# Task 2: Implement a Tool Composition System
# Build a system where tools can be composed -- the output of one tool automatically becomes available as input context for subsequent tool calls. 
# Define a scenario with at least 3 chained tools (e.g., search_product -> check_inventory -> calculate_discount). 
# Implement the composition logic that passes data between tools while maintaining type safety. 
# Add error handling that can recover from a failed intermediate tool call.

import json
from typing import Dict, Any, Callable


def find_product_id(product_name: str) -> dict:
    """Map product name to ID."""
    catalog = {"laptop": "PROD-900", "phone": "PROD-400", "headphones": "PROD-100"}
    name_clean = product_name.lower().strip()
    
    if name_clean not in catalog:
        raise ValueError(f"Product '{product_name}' not found in catalog.")
        
    return {"product_id": catalog[name_clean], "status": "found"}


def check_stock(product_id: str) -> dict:
    """Take product_id, check inventory and price."""
    if product_id == "PROD-400":
        raise ConnectionError("Inventory database timed out.")
        
    inventory = {
        "PROD-900": {"stock": 14, "base_price": 1200.00},
        "PROD-100": {"stock": 0, "base_price": 150.00}
    }
    
    item = inventory.get(product_id)
    if not item:
        raise KeyError(f"Product ID '{product_id}' does not exist in inventory.")
    if item["stock"] <= 0:
        raise ValueError(f"Product ID '{product_id}' is out of stock.")
        
    return {"base_price": item["base_price"], "available": True}


def apply_coupon(base_price: float, coupon_code: str) -> dict:
    """Take price, evaluate and apply discount."""
    coupons = {"SAVE10": 0.10, "SUPER20": 0.20}
    code_clean = coupon_code.upper().strip()
    
    discount_pct = coupons.get(code_clean, 0.0)
    discount_amount = base_price * discount_pct
    final_price = base_price - discount_amount
    
    return {
        "original_price": base_price,
        "discount_applied": f"{discount_pct * 100}%",
        "final_total": round(final_price, 2)
    }



class ToolCompositionEngine:
    def __init__(self):
        self.registry: Dict[str, Callable] = {
            "find_product": find_product_id,
            "check_stock": check_stock,
            "apply_coupon": apply_coupon
        }

    def execute_order_pipeline(self, initial_item: str, coupon: str) -> Dict[str, Any]:
        context: Dict[str, Any] = {"input_item": initial_item, "coupon_code": coupon}
        print(f"\nStarting for: '{initial_item}' with Coupon: '{coupon}'")
        

        print("find_product:")
        res_1 = self.registry["find_product"](context["input_item"])
        context["product_id"] = res_1["product_id"]


        print(f"check_stock (using ID: {context['product_id']}):")
        res_2 = self.registry["check_stock"](context["product_id"])
        context["base_price"] = float(res_2["base_price"])


        print(f" -> Executing: apply_coupon (using Price: ${context['base_price']:.2f})")
        res_3 = self.registry["apply_coupon"](context["base_price"], context["coupon_code"])
        context.update(res_3)


        return {
            "status": "success",
            "product_id": context["product_id"],
            "final_total": context["final_total"],
            "breakdown": {
                "base": context["original_price"],
                "discount": context["discount_applied"]
            }
        }


if __name__ == "__main__":
    engine = ToolCompositionEngine()

    output_a = engine.execute_order_pipeline("laptop", "SUPER20")
    print(f"Result A: {json.dumps(output_a, indent=2)}")

    output_b = engine.execute_order_pipeline("phone", "SAVE10")
    print(f"Result B: {json.dumps(output_b, indent=2)}")