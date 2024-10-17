import openai

context = {
    "Hygiene Products": "HYGI001",
    "Baby Care": "BABY002",
    "Beverages": "BEV003",
    "Snacks": "SNACK004",
    "Dairy Products": "DAIRY005",
    "Frozen Foods": "FROZ006",
    "Cleaning Supplies": "CLEAN007",
    "Personal Care": "PERS008",
    "Condiments and Sauces": "COND009",
    "Cereals": "CEREAL010"
}

def get_grouped_labels(labels):
    question = f"Analyze the list {labels} and group them into categories: {list(context.keys())}. Return only the grouped labels, no need any other info only the reault array with corresponding group and group id."
    openai.api_key = "your key"

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": question}
        ],
        max_tokens=150,
        temperature=0.7
    )
    grouped_output = response.choices[0].message.content
    return grouped_output
