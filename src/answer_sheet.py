generic_instruction: f"Given this context: {context} provide the grower advice on whether the {property} value of {value} is within the optimal range for growing Cannabis.  If it is not, provide advice on how to adjust the value. Use only the information provided in the context."

properties = [
    {
        "property": "pH",
        "value": 0,
        "property_instruction": f"The pH reading on the Mehlic-3 report is {value}.  The ideal pH range for growing Cannabis is 6.8-6.9."
    },
    {
        "property": "Soluble Salts",
        "value": 0,
        "property_instruction": f"The level of soluble salts in the Saturated Paste {value} .  The optimal value is between 1,000 - 1,200.  High levels of soluble salts can cause plants to use extra energy to absorb water. If the concentration of salts is higher outside the plant's roots than inside, it can pull water away from the roots, which is not ideal for plant health. Therefore, it's important to monitor soluble salts because they are indicative of nutrient concentration​​."
    }
]
