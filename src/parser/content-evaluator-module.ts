import { Value } from "@sinclair/typebox/value";
import Decimal from "decimal.js";
import { encodingForModel, Tiktoken } from "js-tiktoken";
import OpenAI from "openai";
import { commentEnum, CommentType } from "../configuration/comment-types";
import configuration from "../configuration/config-reader";
import { OPENAI_API_KEY } from "../configuration/constants";
import {
  ContentEvaluatorConfiguration,
  contentEvaluatorConfigurationType,
} from "../configuration/content-evaluator-config";
import logger from "../helpers/logger";
import { IssueActivity } from "../issue-activity";
import openAiRelevanceResponseSchema, { RelevancesByOpenAi } from "../types/openai-type";
import { GithubCommentScore, Module, Result } from "./processor";

/**
 * Evaluates and rates comments.
 */
export class ContentEvaluatorModule implements Module {
  readonly _configuration: ContentEvaluatorConfiguration | null = configuration.incentives.contentEvaluator;
  readonly _openAi = new OpenAI({
    apiKey: OPENAI_API_KEY,
    ...(this._configuration?.openAi.endpoint && { baseURL: this._configuration.openAi.endpoint }),
  });
  private readonly _fixedRelevances: { [k: string]: number } = {};

  _getEnumValue(key: CommentType) {
    let res = 0;

    key.split("_").forEach((value) => {
      res |= Number(commentEnum[value as keyof typeof commentEnum]);
    });
    return res;
  }

  constructor() {
    if (this._configuration?.multipliers) {
      this._fixedRelevances = this._configuration.multipliers.reduce((acc, curr) => {
        return {
          ...acc,
          [curr.role.reduce((a, b) => this._getEnumValue(b) | a, 0)]: curr.relevance,
        };
      }, {});
    }
  }

  get enabled(): boolean {
    if (!Value.Check(contentEvaluatorConfigurationType, this._configuration)) {
      console.warn("Invalid / missing configuration detected for ContentEvaluatorModule, disabling.");
      return false;
    }
    return true;
  }

  async transform(data: Readonly<IssueActivity>, result: Result) {
    const promises: Promise<GithubCommentScore[]>[] = [];
    const allCommentsUnClean = data.allComments || [];
    const allComments: { id: number; comment: string; author: string }[] = [];
    for (const commentObj of allCommentsUnClean) {
      if (commentObj.user) {
        allComments.push({ id: commentObj.id, comment: commentObj.body || "", author: commentObj.user.login });
      }
    }

    for (const key of Object.keys(result)) {
      const currentElement = result[key];
      const comments = currentElement.comments || [];
      const specificationBody = data.self?.body;
      if (specificationBody && comments.length) {
        promises.push(
          this._processComment(comments, specificationBody, allComments, key).then(
            (commentsWithScore) => (currentElement.comments = commentsWithScore)
          )
        );
      }
    }

    await Promise.all(promises);
    return result;
  }

  async _processComment(
    comments: Readonly<GithubCommentScore>[],
    specificationBody: string,
    allComments: { id: number; comment: string; author: string }[],
    author: string
  ) {
    const commentsWithScore: GithubCommentScore[] = [...comments];
    // exclude comments that have fixed relevance multiplier. e.g. review comments = 1
    const commentsToEvaluate: { id: number; comment: string; author: string }[] = [];
    for (let i = 0; i < commentsWithScore.length; i++) {
      const currentComment = commentsWithScore[i];
      if (!this._fixedRelevances[currentComment.type]) {
        commentsToEvaluate.push({
          id: currentComment.id,
          comment: currentComment.content,
          author: author,
        });
      }
    }

    const relevancesByAI = commentsToEvaluate.length
      ? await this._evaluateComments(specificationBody, commentsToEvaluate, allComments)
      : {};

    if (Object.keys(relevancesByAI).length !== commentsToEvaluate.length) {
      console.error("Relevance / Comment length mismatch! \nWill use 1 as relevance for missing comments.");
    }

    for (let i = 0; i < commentsWithScore.length; i++) {
      const currentComment = commentsWithScore[i];
      let currentRelevance = 1; // For comments not in fixed relevance types and missed by OpenAI evaluation

      if (this._fixedRelevances[currentComment.type]) {
        currentRelevance = this._fixedRelevances[currentComment.type];
      } else if (!isNaN(relevancesByAI[currentComment.id])) {
        currentRelevance = relevancesByAI[currentComment.id];
      }

      const currentReward = new Decimal(currentComment.score?.reward || 0);
      currentComment.score = {
        ...(currentComment.score || {}),
        relevance: new Decimal(currentRelevance).toNumber(),
        reward: currentReward.mul(currentRelevance).toNumber(),
      };
    }

    return commentsWithScore;
  }

  /**
   * Will try to predict the maximum of tokens expected, to a maximum of totalTokenLimit.
   */
  _calculateMaxTokens(prompt: string, totalTokenLimit: number = 16384) {
    const tokenizer: Tiktoken = encodingForModel("gpt-4o-2024-08-06");
    const inputTokens = tokenizer.encode(prompt).length;
    return Math.min(inputTokens, totalTokenLimit);
  }

  _generateDummyResponse(comments: { id: number; comment: string }[]) {
    return comments.reduce<Record<string, number>>((acc, curr) => {
      return { ...acc, [curr.id]: 0.5 };
    }, {});
  }

  async _evaluateComments(
    specification: string,
    comments: { id: number; comment: string; author: string }[],
    allComments: { id: number; comment: string; author: string }[]
  ): Promise<RelevancesByOpenAi> {
    const prompt = this._generatePrompt(specification, allComments, comments);
    const dummyResponse = JSON.stringify(this._generateDummyResponse(comments), null, 2);
    const maxTokens = this._calculateMaxTokens(dummyResponse);

    const response: OpenAI.Chat.ChatCompletion = await this._openAi.chat.completions.create({
      model: this._configuration?.openAi.model || "gpt-4o-2024-08-06",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: prompt,
        },
      ],
      max_tokens: maxTokens,
      top_p: 1,
      temperature: 1,
      frequency_penalty: 0,
      presence_penalty: 0,
    });

    const rawResponse = String(response.choices[0].message.content);
    logger.info(`OpenAI raw response (using max_tokens: ${maxTokens}): ${rawResponse}`);

    const jsonResponse = JSON.parse(rawResponse);

    try {
      const relevances = Value.Decode(openAiRelevanceResponseSchema, jsonResponse);
      logger.info(`Relevances by OpenAI: ${JSON.stringify(relevances)}`);
      return relevances;
    } catch (e) {
      logger.error(`Invalid response type received from openai while evaluating: ${jsonResponse} \n\nError: ${e}`);
      throw new Error("Error in evaluation by OpenAI.");
    }
  }

  _generatePrompt(
    issue: string,
    allComments: { id: number; comment: string; author: string }[],
    comments: { id: number; comment: string; author: string }[]
  ) {
    if (!issue?.length) {
      throw new Error("Issue specification comment is missing or empty");
    }

    return `Please evaluate the relevance of the following GitHub comments in relation to the specified issue. Assess how each comment contributes to clarifying or solving the issue. 

    **Issue Specification:**
    "${issue}"

    **All Comments:**
    ${JSON.stringify(allComments, null, 2)}

    **Comments for Evaluation:**
    ${JSON.stringify(comments, null, 2)}

    For each comment provided for evaluation, assign a relevance score between 0 and 1, where:
    - 1 means the comment is highly relevant and significantly enhances the issue understanding.
    - 0 means the comment is irrelevant or does not contribute any value.

    Respond with a JSON object where each key is the comment ID from the evaluation comments, and the value is a float representing the relevance score. Ensure that the total number of properties in your response matches the number of comments being evaluated: ${comments.length}.`;
  }
}
