class Plugin:
    def __init__(self):
        self.name = "BasePlugin"
        self.description = "Base plugin description."

    def execute(self, text):
        return text


class SentimentAnalyzer(Plugin):
    def __init__(self):
        self.name = "SentimentAnalyzer"
        self.description = "Identifies the overall tone of a given text."

    def execute(self, text):
        cleaned = text.lower()
        if "great" in cleaned or "clean" in cleaned or "helps" in cleaned:
            return "Positive"
        elif "bad" in cleaned or "error" in cleaned or "fail" in cleaned:
            return "Negative"
        return "Neutral"


class TextSummarizer(Plugin):
    def __init__(self):
        self.name = "TextSummarizer"
        self.description = "Trims text to a short summary snippet."

    def execute(self, text):
        if len(text) > 35:
            return text[:35] + "..."
        return text


class CodeReviewer(Plugin):
    def __init__(self):
        self.name = "CodeReviewer"
        self.description = "Checks text for code blocks and programming indicators."

    def execute(self, text):
        if "def " in text or "class " in text or "import " in text:
            return "Contains Code Structures"
        return "Plain Text Structure"

class PluginManager:
    def __init__(self):
        self._plugins = {}

    def register_plugin(self, plugin_instance):
        self._plugins[plugin_instance.name] = plugin_instance

    def list_plugins(self):
        print("Registered Plugins:")
        for name, p in self._plugins.items():
            print(f" - {name}: {p.description}")

    def run_all(self, target_text):
        results = {}
        for name, plugin in self._plugins.items():
            results[name] = plugin.execute(target_text)
        return results


if __name__ == "__main__":
    manager = PluginManager()
    manager.register_plugin(SentimentAnalyzer())
    manager.register_plugin(TextSummarizer())
    manager.register_plugin(CodeReviewer())

    manager.list_plugins()
    
    sample_doc = "Machine learning is a subset of artificial intelligence."
    print(f"\nProcessing Text: '{sample_doc}'")
    
    output_report = manager.run_all(sample_doc)

    for plugin_name, analysis in output_report.items():
        print(f" [{plugin_name}] Result -> {analysis}")