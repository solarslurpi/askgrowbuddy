instruction = """Using only the knowledge provided in the context: \n\n--> {context} -<\n\nprovide the Cannabis grower advice on whether the {property} value of {value} is within this optimal range {optimal_range} for growing Cannabis.  If it is not, provide advice on how to adjust the value using only the information provided in the context. If the context does not contain the information needed, respond with 'I do not have enough information to provide advice.'"""

properties = [
    {
        "name": "pH",
        "value": 0,
        "optimal_range": [6.8, 6.9]
    },
    {
        "name": "soluble_salts_ppm",
        "value": 0,
        "optimal_range": [1000, 1200]
    }

]
