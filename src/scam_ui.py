import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time


class SpamDetectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Spam Detector")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Email Spam Detector",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Email Details Section
        details_frame = ttk.LabelFrame(main_frame, text="Email Details", padding="10")
        details_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        details_frame.columnconfigure(1, weight=1)

        # Sender
        ttk.Label(details_frame, text="Sender:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.sender_entry = ttk.Entry(details_frame, width=50)
        self.sender_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # Recipient (Sendee)
        ttk.Label(details_frame, text="Recipient:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.recipient_entry = ttk.Entry(details_frame, width=50)
        self.recipient_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # Subject
        ttk.Label(details_frame, text="Subject:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.subject_entry = ttk.Entry(details_frame, width=50)
        self.subject_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)

        # Body
        ttk.Label(details_frame, text="Body:").grid(row=3, column=0, sticky=(tk.W, tk.N), pady=5)

        # Text widget with scrollbar for body
        body_frame = ttk.Frame(details_frame)
        body_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0), pady=5)
        body_frame.columnconfigure(0, weight=1)
        body_frame.rowconfigure(0, weight=1)

        self.body_text = tk.Text(body_frame, height=6, width=50, wrap=tk.WORD)
        body_scrollbar = ttk.Scrollbar(body_frame, orient=tk.VERTICAL, command=self.body_text.yview)
        self.body_text.configure(yscrollcommand=body_scrollbar.set)

        self.body_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        body_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Algorithm Selection Section
        algorithm_frame = ttk.LabelFrame(main_frame, text="Detection Algorithms", padding="10")
        algorithm_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(algorithm_frame, text="Primary Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.primary_algorithm = ttk.Combobox(algorithm_frame, values=[
            "Naive Bayes",
            "Decision Tree"
        ], state="readonly", width=25)
        self.primary_algorithm.grid(row=0, column=1, padx=(10, 0), pady=5)
        self.primary_algorithm.set("Naive Bayes")  # Default selection

        ttk.Label(algorithm_frame, text="Secondary Algorithm:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.secondary_algorithm = ttk.Combobox(algorithm_frame, values=[
            "Naive Bayes",
            "Decision Tree"
        ], state="readonly", width=25)
        self.secondary_algorithm.grid(row=1, column=1, padx=(10, 0), pady=5)
        self.secondary_algorithm.set("Support Vector Machine (SVM)")  # Default selection

        # Analysis Button
        self.analyze_button = ttk.Button(main_frame, text="Analyze Email",
                                         command=self.analyze_email, style='Accent.TButton')
        self.analyze_button.grid(row=3, column=0, columnspan=2, pady=20)

        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Spam Detection Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        results_frame.columnconfigure(1, weight=1)

        # Progress bar
        self.progress = ttk.Progressbar(results_frame, mode='indeterminate')
        self.progress.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Spam probability
        ttk.Label(results_frame, text="Spam Probability:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.probability_label = ttk.Label(results_frame, text="---%",
                                           font=('Arial', 14, 'bold'), foreground='blue')
        self.probability_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Classification result
        ttk.Label(results_frame, text="Classification:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.classification_label = ttk.Label(results_frame, text="Not analyzed",
                                              font=('Arial', 12, 'bold'))
        self.classification_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Algorithm details
        self.algorithm_details = ttk.Label(results_frame, text="",
                                           font=('Arial', 9), foreground='gray')
        self.algorithm_details.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

    def validate_input(self):
        """Validate that required fields are filled"""
        if not self.sender_entry.get().strip():
            messagebox.showerror("Input Error", "Please enter the sender's email address.")
            return False
        if not self.subject_entry.get().strip():
            messagebox.showerror("Input Error", "Please enter the email subject.")
            return False
        if not self.body_text.get(1.0, tk.END).strip():
            messagebox.showerror("Input Error", "Please enter the email body.")
            return False
        return True

    def analyze_email(self):
        """Analyze the email for spam"""
        if not self.validate_input():
            return

        # Disable button and start progress
        self.analyze_button.config(state='disabled')
        self.progress.start()
        self.probability_label.config(text="Analyzing...", foreground='blue')
        self.classification_label.config(text="Processing...", foreground='blue')

        # Run analysis in separate thread to keep UI responsive
        threading.Thread(target=self.perform_analysis, daemon=True).start()

    def perform_analysis(self):
        """Simulate spam analysis (replace with your actual algorithm)"""
        # Get input data
        sender = self.sender_entry.get()
        recipient = self.recipient_entry.get()
        subject = self.subject_entry.get()
        body = self.body_text.get(1.0, tk.END)
        primary_algo = self.primary_algorithm.get()
        secondary_algo = self.secondary_algorithm.get()

        # Simulate processing time
        time.sleep(2)

        # TODO: Replace this with your actual spam detection logic
        # This is just a mock implementation
        spam_probability = self.mock_spam_detection(sender, subject, body, primary_algo, secondary_algo)

        # Update UI in main thread
        self.root.after(0, self.update_results, spam_probability, primary_algo, secondary_algo)

    def mock_spam_detection(self, sender, subject, body, primary_algo, secondary_algo):
        """Mock spam detection - replace with your actual implementation"""
        import random

        # Simple mock logic - you'll replace this with your actual algorithms
        spam_keywords = ['free', 'win', 'prize', 'urgent', 'limited time', 'act now',
                         'click here', 'buy now', 'discount', 'offer']

        spam_score = 0
        text_to_check = (subject + ' ' + body).lower()

        for keyword in spam_keywords:
            if keyword in text_to_check:
                spam_score += 15

        # Add some randomness and algorithm-specific adjustments
        if primary_algo == "Naive Bayes":
            spam_score += random.randint(-10, 15)
        elif primary_algo == "Neural Network":
            spam_score += random.randint(-5, 20)

        # Ensure probability is between 0 and 100
        spam_probability = min(100, max(0, spam_score + random.randint(0, 30)))

        return spam_probability

    def update_results(self, spam_probability, primary_algo, secondary_algo):
        """Update the UI with analysis results"""
        # Stop progress bar
        self.progress.stop()

        # Update probability
        self.probability_label.config(text=f"{spam_probability:.1f}%")

        # Update classification and color
        if spam_probability >= 70:
            classification = "SPAM"
            color = 'red'
        elif spam_probability >= 30:
            classification = "SUSPICIOUS"
            color = 'orange'
        else:
            classification = "LEGITIMATE"
            color = 'green'

        self.classification_label.config(text=classification, foreground=color)
        self.probability_label.config(foreground=color)

        # Update algorithm details
        self.algorithm_details.config(text=f"Analysis performed using {primary_algo} and {secondary_algo}")

        # Re-enable button
        self.analyze_button.config(state='normal')


def main():
    root = tk.Tk()
    app = SpamDetectorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
