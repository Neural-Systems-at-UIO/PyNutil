from PyQt6.QtCore import QObject, pyqtSignal
import sys
import io

class TextRedirector(QObject):
    """Redirects text output to a Qt signal."""
    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():  # Only emit non-empty texts
            self.text_written.emit(text)

    def flush(self):
        pass

class LogManager:
    """Manages application logging."""

    def __init__(self, output_widget):
        """
        Initialize the log manager.

        Args:
            output_widget: Widget to display logs
        """
        self.output_widget = output_widget
        self.log_collection = ""
        self.current_progress = ""

    def clear(self):
        """Clear all logs."""
        self.log_collection = ""
        self.current_progress = ""
        self.update_display()

    def append(self, text: str):
        """
        Append text to the log.

        Args:
            text: Text to append
        """
        self.log_collection += text.replace("\n", "<br>") + "<br>"
        self.update_display()

    def set_progress(self, text: str):
        """
        Set the current progress text.

        Args:
            text: Progress text
        """
        self.current_progress = text
        self.update_display()

    def update_display(self):
        """Update the output widget with current logs."""
        self.output_widget.setHtml(self.log_collection + self.current_progress)
        sb = self.output_widget.verticalScrollBar()
        sb.setValue(sb.maximum())
