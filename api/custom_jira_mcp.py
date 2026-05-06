from mcp.server.fastmcp import FastMCP
import os
import requests
import json
import base64
from dotenv import load_dotenv

# Load env variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Initialize FastMCP
mcp = FastMCP("CustomJiraMCP")

def get_auth_headers():
    email = os.environ.get("JIRA_EMAIL")
    token = os.environ.get("JIRA_API_TOKEN")
    
    if not email or not token:
        raise ValueError("Missing JIRA_EMAIL or JIRA_API_TOKEN in environment")
        
    auth_str = f"{email}:{token}"
    b64_auth = base64.b64encode(auth_str.encode("ascii")).decode("ascii")
    
    return {
        "Authorization": f"Basic {b64_auth}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

def get_base_url():
    cloud_id = os.environ.get("ATLASSIAN_CLOUD_ID", "")
    if not cloud_id:
        raise ValueError("Missing ATLASSIAN_CLOUD_ID in environment")
    # Handle bare UUID (e.g. "3971165f-...") vs full URL (e.g. "https://romevela.atlassian.net")
    if not cloud_id.startswith("http"):
        # Try ATLASSIAN_SITE_URL as a fallback, otherwise build from the site name
        site_url = os.environ.get("ATLASSIAN_SITE_URL", "")
        if site_url.startswith("http"):
            return site_url.rstrip("/")
        raise ValueError(
            f"ATLASSIAN_CLOUD_ID looks like a UUID ('{cloud_id}'). "
            "Please set it to your full site URL, e.g. https://yoursite.atlassian.net"
        )
    return cloud_id.rstrip("/")

def _resolve_account_id(query: str) -> str | None:
    """
    Internal helper: look up a Jira accountId by email address OR display name.
    Tries email search first, then falls back to display name matching.
    """
    url = f"{get_base_url()}/rest/api/3/user/search"
    headers = get_auth_headers()
    resp = requests.get(url, headers=headers, params={"query": query})
    if resp.status_code == 200:
        users = resp.json()
        # Exact email match
        for u in users:
            if u.get("emailAddress", "").lower() == query.lower():
                return u.get("accountId")
        # Exact display name match (case-insensitive)
        for u in users:
            if u.get("displayName", "").lower() == query.lower():
                return u.get("accountId")
        # Partial display name match
        for u in users:
            if query.lower() in u.get("displayName", "").lower():
                return u.get("accountId")
        # Last resort: first result
        if users:
            return users[0].get("accountId")
    return None

# Keep old name as alias for backwards compatibility
_resolve_account_id_by_email = _resolve_account_id


@mcp.tool()
def list_jira_users(max_results: int = 100) -> str:
    """
    List all active users in your Jira instance.
    Returns each user's display name, email address, and accountId.
    Use max_results to control how many users to fetch (default: 100).
    """
    url = f"{get_base_url()}/rest/api/3/users/search"
    headers = get_auth_headers()

    all_users = []
    start_at = 0
    batch = min(max_results, 50)

    while len(all_users) < max_results:
        resp = requests.get(
            url,
            headers=headers,
            params={"startAt": start_at, "maxResults": batch},
        )
        if resp.status_code != 200:
            return f"Failed to list users. Status: {resp.status_code}, Response: {resp.text}"

        page = resp.json()
        if not page:
            break

        for u in page:
            # Skip bot/service accounts (no email)
            if u.get("accountType") == "atlassian" or u.get("emailAddress"):
                all_users.append(u)

        if len(page) < batch:
            break  # last page
        start_at += batch

    all_users = all_users[:max_results]

    if not all_users:
        return "No users found in this Jira instance."

    lines = [f"{'Display Name':<30} {'Email':<35} {'AccountId'}"]
    lines.append("-" * 90)
    for u in all_users:
        name = u.get("displayName", "N/A")
        email = u.get("emailAddress", "N/A")
        account_id = u.get("accountId", "N/A")
        lines.append(f"{name:<30} {email:<35} {account_id}")

    return f"Found {len(all_users)} user(s):\n\n" + "\n".join(lines)


@mcp.tool()
def search_jira_users(query: str) -> str:
    """
    Search for Jira users by name OR email address.
    Use this when you know someone's name (e.g. 'python bot', 'Rahul') but not their accountId.
    Returns matching users with display name, email, and accountId.
    """
    url = f"{get_base_url()}/rest/api/3/user/search"
    headers = get_auth_headers()
    resp = requests.get(url, headers=headers, params={"query": query})

    if resp.status_code != 200:
        return f"Failed to search users. Status: {resp.status_code}, Response: {resp.text}"

    users = resp.json()
    if not users:
        return f"No Jira user found matching: '{query}'"

    results = []
    for u in users:
        results.append(
            f"Name: {u.get('displayName')} | Email: {u.get('emailAddress', 'N/A')} | AccountId: {u.get('accountId')}"
        )
    return f"Found {len(results)} match(es) for '{query}':\n" + "\n".join(results)


@mcp.tool()
def get_jira_user_by_email(email: str) -> str:
    """
    Look up a Jira user by their email address.
    Returns the user's display name and accountId.
    """
    return search_jira_users(email)


@mcp.tool()
def create_jira_ticket(
    project_key: str,
    summary: str,
    description: str,
    issue_type: str = "Task",
    assignee_name: str = None,
    assignee_email: str = None,
    assignee_id: str = None,
) -> str:
    """
    Create a new Jira ticket.

    You can assign the ticket by any of:
    - assignee_name:  the user's display name (e.g. 'python bot', 'Rahul') — resolved automatically
    - assignee_email: the user's email address — resolved automatically
    - assignee_id:    the user's Jira accountId directly (takes precedence if provided)
    """
    url = f"{get_base_url()}/rest/api/3/issue"
    headers = get_auth_headers()

    # Resolve assignee: id > email > name
    resolved_id = assignee_id
    assignee_label = assignee_id
    if not resolved_id and assignee_email:
        resolved_id = _resolve_account_id(assignee_email)
        assignee_label = assignee_email
        if not resolved_id:
            return f"Could not find a Jira user with email '{assignee_email}'. Ticket not created."
    if not resolved_id and assignee_name:
        resolved_id = _resolve_account_id(assignee_name)
        assignee_label = assignee_name
        if not resolved_id:
            return f"Could not find a Jira user named '{assignee_name}'. Ticket not created."

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"text": description, "type": "text"}],
                    }
                ],
            },
            "issuetype": {"name": issue_type},
        }
    }

    if resolved_id:
        payload["fields"]["assignee"] = {"id": resolved_id}

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 201:
        data = response.json()
        ticket_key = data.get("key")
        return (
            f"Successfully created Jira ticket: {ticket_key} — "
            f"{get_base_url()}/browse/{ticket_key}"
            + (f" (assigned to {assignee_label})" if resolved_id else "")
        )
    else:
        return f"Failed to create ticket. Status: {response.status_code}, Response: {response.text}"


@mcp.tool()
def add_jira_comment(issue_key: str, comment: str) -> str:
    """
    Add a comment to an existing Jira ticket.
    """
    url = f"{get_base_url()}/rest/api/3/issue/{issue_key}/comment"
    headers = get_auth_headers()
    payload = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"text": comment, "type": "text"}],
                }
            ],
        }
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    if resp.status_code == 201:
        return f"Comment added to {issue_key} successfully."
    return f"Failed to add comment. Status: {resp.status_code}, Response: {resp.text}"


@mcp.tool()
def update_jira_ticket(
    issue_key: str,
    summary: str = None,
    description: str = None,
    assignee_email: str = None,
    assignee_id: str = None,
    status_transition: str = None,
) -> str:
    """
    Update fields on an existing Jira ticket.
    Optionally transition its status (e.g. 'In Progress', 'Done').
    """
    base = get_base_url()
    headers = get_auth_headers()
    update_fields: dict = {}

    if summary:
        update_fields["summary"] = {"set": summary}
    if description:
        update_fields["description"] = {
            "set": {
                "type": "doc",
                "version": 1,
                "content": [
                    {"type": "paragraph", "content": [{"text": description, "type": "text"}]}
                ],
            }
        }

    resolved_id = assignee_id
    if not resolved_id and assignee_email:
        resolved_id = _resolve_account_id_by_email(assignee_email)
        if not resolved_id:
            return f"Could not find a Jira user with email '{assignee_email}'."

    if resolved_id:
        update_fields["assignee"] = {"set": {"id": resolved_id}}

    messages = []
    if update_fields:
        resp = requests.put(
            f"{base}/rest/api/3/issue/{issue_key}",
            headers=headers,
            data=json.dumps({"update": update_fields}),
        )
        if resp.status_code == 204:
            messages.append(f"Fields updated on {issue_key}.")
        else:
            messages.append(f"Field update failed. Status: {resp.status_code}, {resp.text}")

    # Handle status transition
    if status_transition:
        tr_resp = requests.get(f"{base}/rest/api/3/issue/{issue_key}/transitions", headers=headers)
        if tr_resp.status_code == 200:
            transitions = tr_resp.json().get("transitions", [])
            match = next(
                (t for t in transitions if status_transition.lower() in t["name"].lower()), None
            )
            if match:
                requests.post(
                    f"{base}/rest/api/3/issue/{issue_key}/transitions",
                    headers=headers,
                    data=json.dumps({"transition": {"id": match["id"]}}),
                )
                messages.append(f"Status transitioned to '{match['name']}'.")
            else:
                available = [t["name"] for t in transitions]
                messages.append(f"Transition '{status_transition}' not found. Available: {available}")

    return " | ".join(messages) if messages else "Nothing to update."


if __name__ == "__main__":
    mcp.run()
