# # Rasa-Chatbot/rasa_project/actions.py

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests

class ActionCheckProduct(Action):
    def name(self) -> Text:
        return "action_check_product"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        product = tracker.get_slot("product")
        
        response = requests.get(f"https://product-service/api/products/{product}")
        
        if response.status_code == 200:
            data = response.json()
            dispatcher.utter_message(
                text=f"{product} — {data.get('short_description')} Price: ${data.get('price')}"
            )
        else:
            dispatcher.utter_message(text=f"Sorry, I couldn't find information for {product}.")
        return []


class ActionCreateOrder(Action):
    def name(self) -> Text:
        return "action_create_order"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        product = tracker.get_slot('product')
        
        order_id = "ORD123456"  # simulated
        dispatcher.utter_message(text=f"Your order has been placed. Order ID: {order_id}")
        return []
