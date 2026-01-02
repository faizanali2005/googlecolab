import heapq
from typing import Dict, Any, List, Tuple, Optional

# ===================================================================
# 1. DUMMY DATA (COMBINED INTO SINGLE FILE)
# ===================================================================

DUMMY_DATA = {
    "restaurants": {
        "R1_Italian": {"cuisine": "Italian", "location": "Downtown", "rating": 4.5, "hours": "9-22"},
        "R2_Mexican": {"cuisine": "Mexican", "location": "Suburbs", "rating": 3.8, "hours": "11-21"},
        "R3_Sushi": {"cuisine": "Japanese", "location": "Downtown", "rating": 4.9, "hours": "17-23"},
        "R4_Vegan": {"cuisine": "Vegan", "location": "Uptown", "rating": 4.2, "hours": "8-18"},
    },
    "menu_items": {
        "Pizza_Margherita": {"restaurant": "R1_Italian", "cost": 15, "prep_time": 15},
        "Taco_Combo": {"restaurant": "R2_Mexican", "cost": 12, "prep_time": 10},
        "Salmon_Roll": {"restaurant": "R3_Sushi", "cost": 28, "prep_time": 20},
        "Tofu_Curry": {"restaurant": "R4_Vegan", "cost": 18, "prep_time": 25},
        "Lasagna": {"restaurant": "R1_Italian", "cost": 22, "prep_time": 30},
    }
}

CUSTOMER_PREFERENCES = {
    "max_budget": 30,
    "preferred_cuisine": "Italian",
    "time_window": (19, 21)  # 7 PM to 9 PM
}


# ===================================================================
# 2. CSP SETUP AND HEURISTICS
# ===================================================================

class RestaurantCSP:
    def __init__(self, data, prefs):
        self.restaurants = data["restaurants"]
        self.menu_items = data["menu_items"]
        self.prefs = prefs

        # Variables: Menu Item IDs
        self.variables = list(self.menu_items.keys())

        # Goal Anchor: Find the restaurant ID matching the preferred cuisine for the heuristic
        self.goal_anchor_r_id = next((r_id for r_id, details in self.restaurants.items()
                                      if details['cuisine'] == prefs['preferred_cuisine']), None)

    def check_constraints(self, assignment: Dict[str, str]) -> bool:
        """Checks if a partial or full assignment satisfies global constraints."""

        if not assignment:
            return True

        total_cost = 0

        # Gather restaurant IDs associated with chosen items
        restaurant_ids_in_choice = set()

        for item_name in assignment.keys():
            cost = self.menu_items[item_name]['cost']
            total_cost += cost
            restaurant_ids_in_choice.add(self.menu_items[item_name]['restaurant'])

        # Constraint 1: Budget
        if total_cost > self.prefs['max_budget']:
            return False

        # Constraint 2: Consistency (All items must come from the same restaurant)
        if len(restaurant_ids_in_choice) > 1:
            return False

        # Constraint 3: Cuisine Match (If we chose items, they must match the preference)
        if restaurant_ids_in_choice:
            r_id = list(restaurant_ids_in_choice)[0]
            rest_details = self.restaurants[r_id]

            if rest_details['cuisine'] != self.prefs['preferred_cuisine']:
                return False

        return True

    def select_unassigned_variable(self, assignment: Dict[str, str]) -> Optional[str]:
        """
        MRV Heuristic (Simplified): Selects the next item to potentially add to the set.
        We just pick the first available item that hasn't been added yet.
        """
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None

        return unassigned[0]


def h_cost(state_items_set: frozenset, goal_r_id: str, restaurant_data: Dict, menu_data: Dict) -> float:
    """
    Heuristic function H(n): Estimated cost from current state to goal.
    We estimate remaining cost based on how far the current choice is from the ideal restaurant.
    """
    if not state_items_set:
        # Initial state: High uncertainty, default penalty.
        return 50.0

        # Determine the restaurant associated with the current choices (all should be the same restaurant)
    first_item_id = list(state_items_set)[0]
    current_r_id = menu_data[first_item_id]['restaurant']

    rating_value = restaurant_data[current_r_id]['rating']

    penalty = 0.0

    # Penalize if we are not at the ideal restaurant anchor
    if goal_r_id and current_r_id != goal_r_id:
        penalty = 100.0  # Huge penalty for being the wrong type of restaurant

    # H(n) should estimate *remaining* cost. High rating means path is good (low remaining cost estimate).
    # We use (5.0 - rating) as a measure of 'badness' to add to the heuristic estimate.
    return (5.0 - rating_value) * 5.0 + penalty


# ===================================================================
# 3. A* SEARCH WITH FORWARD CHECKING & BOUNDING
# ===================================================================

def a_star_search(csp: RestaurantCSP, alpha: float = float('inf')):
    # State in PQ: (f_cost, g_cost, state_tuple)
    # State Tuple: (frozenset_of_items_chosen, last_item_chosen)

    goal_r_id = csp.goal_anchor_r_id

    # Initial State
    initial_state_tuple = (frozenset(), None)
    initial_g = 0
    initial_h = h_cost(initial_state_tuple[0], goal_r_id, csp.restaurants, csp.menu_items)
    initial_f = initial_g + initial_h

    pq = [(initial_f, initial_g, initial_state_tuple)]

    # g_costs: Stores the minimum cost found to reach a specific state (set of items)
    g_costs = {initial_state_tuple: initial_g}

    # Alpha represents the lowest complete, valid cost found so far (our bound/pruning mechanism)
    current_alpha = alpha
    best_solution = None

    while pq:
        f, g, current_state_tuple = heapq.heappop(pq)
        current_items_set, _ = current_state_tuple

        # --- Game Theory / Alpha Bounding Check (Pruning) ---
        # If the current path cost G already exceeds the best known full solution cost (Alpha), prune.
        if g >= current_alpha:
            continue

            # --- Goal Test ---
        # A state is considered a 'goal' if it is fully constrained AND valid.
        # Since we are searching for the *smallest valid set*, we check constraints here.
        current_assignment = {item: csp.menu_items[item]['restaurant'] for item in current_items_set}

        if csp.check_constraints(current_assignment) and len(current_items_set) > 0:

            # Found a valid, complete path. Update Alpha (the best solution found so far).
            if g < current_alpha:
                current_alpha = g
                best_solution = {
                    "path_cost_g": g,
                    "assignment": current_assignment,
                    "total_cost": g,
                    "restaurant_id": list(current_assignment.values())[0]
                }

            # Do not stop immediately; continue to ensure we find the global minimum (A*'s nature)
            continue

        # --- Expansion: Try adding the next unassigned item (simulated next move) ---

        next_item_to_add = csp.select_unassigned_variable(current_assignment)

        if next_item_to_add is None:
            continue  # No more items to add to this path

        # Try adding *this* potential next item:
        next_item = next_item_to_add

        new_items_set = current_items_set | {next_item}
        new_assignment = {item: csp.menu_items[item]['restaurant'] for item in new_items_set}

        # --- Forward Checking ---
        if csp.check_constraints(new_assignment):

            new_g = g + csp.menu_items[next_item]['cost']
            new_state_tuple = (frozenset(new_items_set), next_item)

            # Prune based on the most recently updated Alpha
            if new_g >= current_alpha:
                continue

                # Standard A* state check
            if new_state_tuple not in g_costs or new_g < g_costs[new_state_tuple]:
                g_costs[new_state_tuple] = new_g

                new_h = h_cost(new_state_tuple[0], goal_r_id, csp.restaurants, csp.menu_items)
                new_f = new_g + new_h

                heapq.heappush(pq, (new_f, new_g, new_state_tuple))

    return best_solution


# ===================================================================
# 4. CONCEPTUAL GAME TREE AND OUTPUT
# ===================================================================

def conceptual_game_tree_and_strategy(result: Optional[Dict]):
    print("\n--- Game Theory & Optimal Strategy (Conceptual) ---")

    if not result:
        print("Optimal Strategy: No solution found. Customer constraints are too strict.")
        return

    r_id = result['restaurant_id']
    restaurant_details = DUMMY_DATA['restaurants'][r_id]

    strategy = f"""
    Optimal Strategy Found: 
    The search minimized the cost G while enforcing constraints (Forward Checking) 
    and used the heuristic H (rating/cuisine proximity) to guide the search. 
    The 'Alpha' bound ensured we pruned paths worse than the best solution found so far.

    1. Restaurant Chosen: {restaurant_details['location']} ({restaurant_details['cuisine']})
    2. Menu Choices: {list(result['assignment'].keys())}
    3. Total Cost (Optimal G): ${result['total_cost']}

    This selection represents the Minimax-optimal choice under the 'Max Utility' objective, 
    where utility is inversely proportional to cost and deviation from preference.
    """
    print(strategy)


# --- Main Execution ---
if __name__ == "__main__":

    print("--- Restaurant Recommendation System (CSP + A* + Game Pruning) ---")

    # 1. Setup CSP
    csp = RestaurantCSP(DUMMY_DATA, CUSTOMER_PREFERENCES)
    print(
        f"Target Cuisine: {CUSTOMER_PREFERENCES['preferred_cuisine']} | Max Budget: ${CUSTOMER_PREFERENCES['max_budget']}")

    # 2. Run A* Search (Initialize Alpha to Infinity)
    final_result = a_star_search(csp, alpha=float('inf'))

    # 3. Output Results
    print("\n--- A* Search Final Result ---")
    if final_result:
        print(f"SUCCESS! Lowest Path Cost (G): {final_result['total_cost']}")
        print(f"Selected Items: {list(final_result['assignment'].keys())}")
        print(f"Restaurant: {DUMMY_DATA['restaurants'][final_result['restaurant_id']]['cuisine']}")

        # 4. Game Theory Analysis
        conceptual_game_tree_and_strategy(final_result)

    else:
        print("Search completed, but no solution satisfied ALL hard constraints.")
