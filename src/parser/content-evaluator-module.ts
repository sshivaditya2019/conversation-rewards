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

    return `Instruction: 
    Go through all the comments first keep them in memory, then start with the following prompt

    OUTPUT FORMAT:
    {ID: CONNECTION SCORE} For Each record, based on the average value from the CONNECTION SCORE from ALL COMMENTS, TITLE and BODY, one for each comment under evaluation
    Global Context:
    Specification
    "${issue}"
    ALL COMMENTS:
    ${JSON.stringify(allComments, null, 2)}
    IMPORTANT CONTEXT:
    You have now seen all the comments made by other users, keeping the comments in mind think in what ways comments to be evaluated be connected. The comments that were related to the comment under evaluation might come after or before them in the list of all comments but they would be there in ALL COMMENTS. COULD BE BEFORE OR AFTER, you have diligently search through all the comments in ALL COMMENTS.
    START EVALUATING:
    ${JSON.stringify(comments, null, 2)}
    POST EVALUATION:
    THE RESULT FROM THIS SHOULD BE ONLY THE SCORES BASED ON THE FLOATING POINT VALUE CONNECTING HOW CLOSE THE COMMENT IS FROM ALL COMMENTS AND TITLE AND BODY.

    Now Assign them scores a float value ranging from 0 to 1, where 0 is spam (lowest value), and 1 is something that's very relevant (Highest Value), here relevance should mean a variety of things, it could be a fix to the issue, it could be a bug in solution, it could a SPAM message, it could be comment, that on its own does not carry weight, but when CHECKED IN ALL COMMENTS, may be a crucial piece of information for debugging and solving the ticket. If YOU THINK ITS NOT RELEATED to ALL COMMENTS or TITLE OR ISSUE SPEC, then give it a 0 SCORE.

    OUTPUT:
    RETURN ONLY A JSON with the ID and the connection score (FLOATING POINT VALUE) with ALL COMMENTS TITLE AND BODY for each comment under evaluation.  RETURN ONLY ONE CONNECTION SCORE VALUE for each comment;
    }`;
  }
}
