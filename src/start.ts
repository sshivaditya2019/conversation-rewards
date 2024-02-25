import { getOctokitInstance } from "./get-authentication-token";
import { GitHubIssue, GitHubPullRequest, GitHubTimelineEvent, GitHubUser } from "./github-types";

// async function main(gitHubIssueUrl: GitHubIssue["html_url"]) {
// const issueParams = parseGitHubUrl(gitHubIssueUrl);
// const issue = await getIssue(issueParams);
// const pullRequestLinks = await collectLinkedPulls(issueParams);
// const pullRequestDetails = await Promise.all(
//   pullRequestLinks.map((link) =>
//     getPullRequest({
//       owner: link.source.issue.repository.owner.login,
//       repo: link.source.issue.repository.name,
//       pull_number: link.source.issue.number,
//     })
//   )
// );

// const users = await getTimelineUsers(issueParams);
// const users = getUsers(issueParams, pullRequestDetails);

// const usersByType = {
//   assignees: users.filter((user) => user.isAssignee),
//   authors: users.filter((user) => user.isAuthor),
//   collaborators: users.filter((user) => user.isCollaborator),
//   remainder: users.filter((user) => user.isRemainder),
// };

/**
   * gather context
   * * this includes:

   * * * the github issue,
  * * * * comments

  * * * the linked pull request
  * * * * comments
  * * * * commits
  * * * * reviews
  * * * * review comments

  * * * the users
   * * * * isAssignee?
   * * * * isAuthor?
   * * * * isCollaborator?
   * * * * isRemainder?
   */
// }

export async function getIssue(params: IssueParams): Promise<GitHubIssue> {
  const octokit = getOctokitInstance();
  return (await octokit.issues.get(params)).data;
}

export async function getIssueEvents(issueParams: IssueParams) {
  const octokit = getOctokitInstance();
  return await octokit.paginate(octokit.issues.listEvents.endpoint.merge(issueParams));
}

export async function getIssueComments(issueParams: IssueParams) {
  const octokit = getOctokitInstance();
  return await octokit.paginate(octokit.issues.listComments.endpoint.merge(issueParams));
}

export async function getAllIssueActivity(issueParams: IssueParams) {
  // @DEV: this is very useful for seeing every type of event that has occurred on the issue
  // in chronological order
  const octokit = getOctokitInstance();
  const [issue, events, comments] = await Promise.all([
    octokit.issues.get(issueParams),
    octokit.paginate(octokit.issues.listEvents.endpoint.merge(issueParams)),
    octokit.paginate(octokit.issues.listComments.endpoint.merge(issueParams)),
  ]);

  const mixedEventsAndComments = [...events, ...comments];
  mixedEventsAndComments.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
  // Prepend the issue to the events array
  mixedEventsAndComments.unshift(issue.data);
  return mixedEventsAndComments;
}

export async function getTimelineUsers(issueParams: IssueParams): Promise<GitHubUser[]> {
  const timelineEvents = await getAllTimelineEvents(issueParams);
  const users = timelineEvents.filter((event) => event.actor).map((event) => event.actor);
  return [...new Set(users)];
}

export async function getAllTimelineEvents(issueParams: IssueParams): Promise<GitHubTimelineEvent[]> {
  const octokit = getOctokitInstance();
  const options = octokit.issues.listEventsForTimeline.endpoint.merge(issueParams);
  return await octokit.paginate(options);
}

export function parseGitHubUrl(url: string): { owner: string; repo: string; issue_number: number } {
  const path = new URL(url).pathname.split("/");
  return {
    owner: path[1],
    repo: path[2],
    issue_number: Number(path[4]),
  };
}

export type IssueParams = ReturnType<typeof parseGitHubUrl>;
type PullParams = { owner: string; repo: string; pull_number: number };

export async function getPullRequest(pullParams: PullParams): Promise<GitHubPullRequest> {
  const octokit = getOctokitInstance();
  return (await octokit.pulls.get(pullParams)).data;
}
