# constants.py
class Texts:
    def __init__(self):
        self.texts = {
            "page_title": "VISIONDECK PRO",
            "title": "VISIONDECK PRO - Card Recognition System",
            "subtitle": "Playing Card Detection",
                "take_snapshot": "Take Snapshot",
                "capturing_cards": "Capturing cards... Please wait.",
                "cards_detected": "Cards detected successfully!",
                "no_cards_detected": "No valid cards detected. Please try again.",
            "detected_cards": "Detected Cards:",
            "card_count": "Total Cards Detected:",
            "clear_results": "Clear Results",
        }

    def get(self, key):
        return self.texts.get(key, key)
