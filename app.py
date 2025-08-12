import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Task Auto-Assignment System", page_icon="📋", layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------
def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def compute_score(last_task, candidate):
    # Base cosine similarity
    score = cosine_similarity(last_task["vector"], candidate["vector"])
    # Continuity bonus
    if candidate["product"] == last_task["product"]:
        score += 0.3
    if candidate["task_id"] == last_task["task_id"]:
        score += 0.5
    return score

def format_time(minutes):
    hour = 8 + (minutes // 60)
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"

def check_requirements_met(task, inventory, required_qty):
    if not task["requirements"]:
        return True
    for req in task["requirements"]:
        if inventory[req] < required_qty:
            return False
    return True

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    if os.path.exists("workers.csv"):
        workers_df = pd.read_csv("workers.csv")
    else:
        workers_df = pd.DataFrame(columns=["Worker"])
        workers_df.to_csv("workers.csv", index=False)

    if os.path.exists("products.csv"):
        products_df = pd.read_csv("products.csv")
    else:
        products_df = pd.DataFrame(columns=[
            "Product","Task","Result","Requirements","Bending","Gluing","Assembling",
            "EdgeScrap","OpenPaper","QualityControl","TimePerPieceSeconds"
        ])
        products_df.to_csv("products.csv", index=False)

    return workers_df, products_df

# -----------------------------
# Scheduling Logic
# -----------------------------
def assign_tasks(products_to_produce, workers, products_df, slot_duration_minutes=30):
    slot_duration_seconds = slot_duration_minutes * 60
    workday_minutes = 8 * 60

    # Build tasks list
    all_task_instances = []
    for product, qty in products_to_produce.items():
        tasks = products_df[products_df["Product"] == product].sort_values(by="Result")
        for _, row in tasks.iterrows():
            requirements = [] if pd.isna(row["Requirements"]) else [r.strip() for r in str(row["Requirements"]).split(",") if r.strip()]
            vector = [row["Bending"], row["Gluing"], row["Assembling"], row["EdgeScrap"], row["OpenPaper"], row["QualityControl"]]
            all_task_instances.append({
                "task_id": row["Result"],
                "description": row["Task"],
                "product": row["Product"],
                "requirements": requirements,
                "vector": vector,
                "time_per_piece": int(row["TimePerPieceSeconds"]),
                "remaining_qty": qty
            })

    current_time_minutes = 0
    current_day = 1
    inventory = defaultdict(int)
    schedule = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    simulation_log = []
    worker_last_task = {w: None for w in workers}

    # Identify true OpenPaper tasks for first slot
    open_paper_tasks = [
        t for t in all_task_instances
        if t["vector"][4] >= 80  # High OpenPaper
        and sum(t["vector"][:4]) < 100  # Low bending, gluing, assembling, edge scrap
        and len(t["requirements"]) == 0  # No prerequisites
    ]

    # Main loop: run until all tasks complete
    loop_counter = 0
    while any(t["remaining_qty"] > 0 for t in all_task_instances):
        loop_counter += 1
        if loop_counter > 30 * 16:  # Failsafe: 30 days of 16 slots
            st.error("Simulation stopped due to exceeding 30 workdays. Check for circular dependencies.")
            break

        current_day = (current_time_minutes // workday_minutes) + 1
        current_slot = (current_time_minutes % workday_minutes) // slot_duration_minutes

        # Available tasks (dependencies met)
        available_tasks = []
        for t in all_task_instances:
            if t["remaining_qty"] > 0:
                if check_requirements_met(t, inventory, products_to_produce[t["product"]]):
                    available_tasks.append(t)

        if not available_tasks:
            current_time_minutes += slot_duration_minutes
            continue

        worker_assignments = {}

        # First slot logic
        if current_time_minutes == 0 and open_paper_tasks:
            for w in workers:
                if open_paper_tasks:
                    worker_assignments[w] = open_paper_tasks.pop(0)
        else:
            for w in workers:
                last_task = worker_last_task[w]
                if last_task:
                    # Pick best based on score (similarity + continuity bonus), tie-break by remaining qty
                    best_task = max(
                        available_tasks,
                        key=lambda t: (compute_score(last_task, t), t["remaining_qty"])
                    )
                else:
                    # First task after prep phase: pick biggest remaining qty
                    best_task = max(available_tasks, key=lambda t: t["remaining_qty"])
                worker_assignments[w] = best_task

        # Process assignments
        for w, task in worker_assignments.items():
            time_remaining = slot_duration_seconds
            dominant_task = task["task_id"]
            pieces_total = 0

            while time_remaining > 0 and task:
                if task["remaining_qty"] <= 0:
                    available_tasks = [t for t in available_tasks if t["remaining_qty"] > 0]
                    if not available_tasks:
                        break
                    task = max(available_tasks, key=lambda t: t["remaining_qty"])
                    continue

                tpp = task["time_per_piece"]
                max_pieces = min(task["remaining_qty"], time_remaining // tpp)
                if max_pieces > 0:
                    task["remaining_qty"] -= max_pieces
                    inventory[task["task_id"]] += max_pieces
                    pieces_total += max_pieces
                    time_remaining -= max_pieces * tpp

                    simulation_log.append({
                        "time": format_time(current_time_minutes),
                        "event": f"Worker {w} produced {max_pieces} pcs of {task['task_id']} ({task['description']})"
                    })
                else:
                    break

            schedule[current_day][w][current_slot] = f"[{dominant_task}] {task['description']} ({pieces_total} pcs)"
            worker_last_task[w] = task

        current_time_minutes += slot_duration_minutes

    # Debug: Check for unfinished tasks
    unfinished = [t for t in all_task_instances if t["remaining_qty"] > 0]
    if unfinished:
        st.warning("Some tasks were not completed:")
        for t in unfinished:
            st.write(f"{t['task_id']} ({t['description']}) - Remaining: {t['remaining_qty']} pcs")

    return {
        "schedule": schedule,
        "inventory": dict(inventory),
        "simulation_log": simulation_log,
        "estimated_days": current_day
    }

# -----------------------------
# Display
# -----------------------------
def display_schedule(schedule, estimated_days):
    st.subheader("Schedule")
    tabs = st.tabs([f"Day {d}" for d in range(1, estimated_days + 1)])
    for i, day in enumerate(range(1, estimated_days + 1)):
        with tabs[i]:
            if day in schedule:
                rows = []
                for slot in range(16):
                    row = {"TIME": format_time(slot * 30)}
                    for w in sorted(schedule[day].keys()):
                        row[w] = schedule[day][w].get(slot, "idle")
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# -----------------------------
# Main App
# -----------------------------
def main():
    workers_df, products_df = load_data()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home","Products","Workers","Production Order"])
    if page == "Home":
        st.header("Welcome")
        st.write("Cosine similarity + continuity-based scheduling with strict prerequisites.")
    elif page == "Products":
        st.header("Product Database")
        st.dataframe(products_df, use_container_width=True)
    elif page == "Workers":
        st.header("Worker Database")
        st.dataframe(workers_df, use_container_width=True)
    elif page == "Production Order":
        st.header("Production Order")

        # Initialize session state for product quantities
        if "products_to_produce" not in st.session_state:
            st.session_state["products_to_produce"] = {}

        # Quick add button
#        if st.button("Add 100 pcs to ALL products"):
#            for product in products_df["Product"].unique():
#                st.session_state["products_to_produce"][product] = 100
#            st.success("✅ All products set to 100 pcs.")

        # Render inputs with persisted values
        for product in products_df["Product"].unique():
            default_qty = st.session_state["products_to_produce"].get(product, 0)
            qty = st.number_input(f"{product}", min_value=0, max_value=1000, value=default_qty, step=1)
            if qty > 0:
                st.session_state["products_to_produce"][product] = qty
            elif product in st.session_state["products_to_produce"]:
                del st.session_state["products_to_produce"][product]

        selected_workers = st.multiselect("Choose Workers", workers_df["Worker"].tolist(), default=workers_df["Worker"].tolist())

        if st.session_state["products_to_produce"] and st.button("🚀 Run Simulation"):
            result = assign_tasks(st.session_state["products_to_produce"], selected_workers, products_df)
            if result:
                st.success(f"Simulation completed! Estimated {result['estimated_days']} day(s).")
                display_schedule(result["schedule"], result["estimated_days"])
                st.subheader("Simulation Log")
                st.dataframe(pd.DataFrame(result["simulation_log"]), use_container_width=True)

if __name__ == "__main__":
    main()


