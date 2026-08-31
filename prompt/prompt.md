

**[https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier)**

That repo (34.7k stars) is what backs the `claude-plugins-official` marketplace your config references — it has a top-level `plugins/` folder for Anthropic's own plugins (code-simplifier is one of ~39 in there) and an `external_plugins/` folder for third-party ones. The `.claude-plugin/marketplace.json` in that repo is the manifest that maps plugin names to their source paths.

If you'd rather browse it locally instead of on GitHub, Claude Code caches installed plugins under `~/.claude/plugins/` after install — you can poke around there too, though GitHub will show you the canonical source (agent definition, `plugin.json`, any skills/commands it ships).

❯ @"code-simplifier:code-simplifier (agent)" After current non-trivial refactor(un-committed changes), review the changed files (ignore tests\\ and its sub folders) before considering the task done.
