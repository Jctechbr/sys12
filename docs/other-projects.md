# Use System One from another MCP server

System One picks a label, answers yes/no, or scores a rubric. It does not know your tools. Your server owns the one-line descriptions. The Cursor chat makes both calls. No tool function inside your server calls System One.

The decision tool lives on the `sys12` MCP server as `sys12_decide`. Read `sys12`'s schema first, then call it.

Copy the NetOps shape. NetOps stores its lines in `tools/core/decision_criteria.json` and serves them with `netops_get_decision_criteria`.

## 1. Add a criteria file in your repository

```json
{
  "descriptions": {
    "your_tool_name": "what this tool actually returns, in one line"
  },
  "context_hints": [
    {
      "match": ["phrase from the question", "second phrase"],
      "hint": "A fact the decision needs that is not a description of one label."
    }
  ]
}
```

The key is the exact label the chat will pass to `sys12_decide`. That is the tool name, or a choice label that is not a tool (`none`, a handoff name, a fault class).

Write a line only when two labels can be confused. A vague line, or a raw docstring, makes the neighbor look plausible and pulls the confidence score down.

A useful line names what the tool returns: the command, the record, or the change it makes. A tool that changes state says it changes configuration and requires confirmation, so a read decision does not select it.

Describe a neighbor by what that neighbor is. Repeating the winner’s subject on the loser (“not RADIUS”, “not liveness”) keeps probability on the loser.

A context hint is returned only when every `match` phrase appears in the question. Use it for a fact that is not a label description, such as a command that is invalid on a given software release.

Leave a label out until you can separate it from the one it gets confused with. Unknown labels are listed in `missing` and are not given an invented line.

Keep MAC addresses, IP addresses, hostnames, and secrets out of this file.

## 2. Expose one read-only tool

Name it `<server>_get_decision_criteria`. Arguments are `labels` (list of strings) and `question` (string, optional). Return this object and nothing else:

| Field | Meaning |
|---|---|
| `notes` | Map of label to the one stored line. Only labels that have a line. |
| `context_hints` | Hint strings whose `match` phrases all appear in `question`. |
| `missing` | Requested labels that have no stored line. |

Reject a blank label. Skip duplicate labels. This tool does not decide and does not call `sys12_decide`.

Register it on every launch of the server, including reduced profiles. A decision can ask about a tool that is not loaded in the current profile, and the description still has to be available.

Do not put these lines in the tool docstrings. Docstrings are sent to the chat with the whole tool list. The lookup is sent only for the labels in the current decision.

## 3. Tell the chat the tool name

The criteria tool does nothing until the chat is told to call it. Add a short paragraph to that project’s skill:

> When the options are this server’s tool names or other labels it describes, call `<server>_get_decision_criteria` first, with those labels and the question. Pass the returned `notes` to `sys12_decide`. Append each `context_hints` line to `context`. The lines live in this repository. Do not write them into the sys12 project.

The System One skill already says: for any server that returns `notes`, `context_hints`, and `missing`, call that server’s criteria tool before `sys12_decide`. The project skill is what names the tool.

## 4. Make the decision call

The chat calls `sys12_decide` on the `sys12` MCP server.

| Argument | Use |
|---|---|
| `question` | The decision, one sentence. |
| `kind` | `choice`, `yes_no`, or `score`. |
| `options` | Exact labels for `choice`. Rubric from low to high for `score`. Omit for `yes_no`. |
| `context` | Facts only, plus any `context_hints` from step 2. |
| `notes` | The `notes` map from step 2. Optional extra lines for labels that came back in `missing`. |

`notes` applies to that call only. System One does not write it to disk. The next call without `notes` sees the bare label.

Redact MAC addresses, IP addresses, and hostnames before they go into `question`, `context`, or `notes`.

## 5. Use the result

Use `answer` only when `escalate` is false.

- Choice and score: confidence below 0.55 escalates. Confidence of exactly 0.55 stands.
- Yes/no: `yes_probability` inside 0.10–0.90 escalates. Exactly 0.10 or 0.90 stands. The reported yes/no still flips at 0.5 so the hint is visible.

When `escalate` is true, the chat decides and ignores `answer`. System One does not call the chat back.

The result is a label or a score. It is not permission to run the chosen tool, push a change, or skip a write gate.

## 6. Correct a miss in your repository

When the chat’s label is right and Clef’s label is wrong, edit the line in your criteria file. Say what the right tool returns, and rewrite the neighbor so it no longer shares the winner’s subject.

Ask the same question again. Keep the edit only if the repeat returns the right label with confidence at or above 0.55. A higher confidence on the same wrong label is not a reason to keep an edit.

Do not save the correction in the sys12 project.
