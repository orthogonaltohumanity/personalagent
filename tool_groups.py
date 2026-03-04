from tools import build_tool_registry, available_functions
from providers import build_tool_schemas

# Ensure the tool registry is populated
build_tool_registry()

TOOL_GROUPS = {
    "web_search": {
        "description": "Search the web and download files",
        "tools": ["search_web", "search_and_download_files"]
    },
    "social_media": {
        "description": "Interact with Moltbook social media platform",
        "tools": [
            "read_social_media_feed", "create_social_media_post",
            "post_file_to_social_media", "create_social_media_comment",
            "social_media_upvote",
            "social_media_downvote", "social_media_upvote_comment",
            "get_social_media_post", "list_community_posts",
            "social_media_search", "get_personal_history",
            "get_user_profile", "list_communities",
            "check_agent_status", "update_profile", "make_community"
        ]
    },
    "document_processing": {
        "description": "Ingest, query, and manage PDFs and CSVs",
        "tools": ["ingest_pdf", "ingest_csv", "query_documents", "list_downloaded_files"]
    },
    "file_operations": {
        "description": "Read and write files in the working directory",
        "tools": ["read_file", "edit"]
    },
    "code_generation": {
        "description": "Generate or edit code using AI code models",
        "tools": ["generate_code", "generate_code_edit"]
    },
    "text_generation": {
        "description": "Generate or edit written text; prefer source-grounded writing (write_text_from_source) and use edit_text for revisions",
        "tools": ["write_text", "edit_text", "write_text_from_source"]
    },
    "version_control": {
        "description": "Git operations in the working directory",
        "tools": [
            "git_init", "git_status", "git_add", "git_commit",
            "git_log", "git_diff", "git_diff_staged",
            "git_branch", "git_checkout", "git_list_branches"
        ]
    },
    "memory": {
        "description": "Persistent memory: save, search, and manage knowledge",
        "tools": [
            "search_memory", "save_memory", "open_memory",
            "edit_memory", "delete_memory", "list_memory_keys",
            "memory_stats", "set_short_term_goal"
        ]
    },
    "system": {
        "description": "Custom tool creation, user interaction, and meta-tools",
        "tools": [
            "create_tool", "list_custom_tools", "remove_custom_tool", "check_in"
        ]
    }
}


def get_group_summary(include_tools=True):
    """Formatted string of all groups with descriptions.

    If include_tools=False, omits individual tool names to prevent the
    planner from hallucinating direct tool calls for executor-only tools.
    """
    lines = []
    for name, group in TOOL_GROUPS.items():
        if include_tools:
            tool_names = ", ".join(group["tools"])
            lines.append(f"  {name}: {group['description']} [{tool_names}]")
        else:
            lines.append(f"  {name}: {group['description']}")
    return "Available tool groups:\n" + "\n".join(lines)


def get_tools_in_group(group_name):
    """Returns list of tool function objects for a given group."""
    if group_name not in TOOL_GROUPS:
        return []
    return [
        available_functions[name]
        for name in TOOL_GROUPS[group_name]["tools"]
        if name in available_functions
    ]


def get_group_tool_schemas(group_name):
    """Returns Ollama tool schemas for just the tools in a group."""
    funcs = get_tools_in_group(group_name)
    return build_tool_schemas(funcs)


def get_group_names():
    """Returns list of all group names."""
    return list(TOOL_GROUPS.keys())


def get_group_description(group_name):
    """Returns the description of a tool group."""
    if group_name not in TOOL_GROUPS:
        return f"Unknown group: {group_name}"
    return TOOL_GROUPS[group_name]["description"]
