

min_pH = 6.8
max_pH = 6.9
pH = m3_report.pH

if pH < min_pH:
    print("write a min pH prompt")
elif pH > max_pH:
    print("write a max pH prompt")

- get_nodes from chromadb using llamaindex retriever
- write the prompt
f"""You are a helpful assistant that analyzes mehlic-3 and saturated paste soil tests.  You focus on tests coming from Cannibis growers who grow in large containers that use a mixture of peat moss, compost, and perlite as their growing medium.  This soiless media is full of microbes that work with organic matter to provide nutrients to the plants.  The Cannibus grower comes to you with the pH value of their soil.  Your job is to write a reply to the grower with advice on what the pH value means and what to do about it.  You know the ideal pH for growing Cannabis in this soiless media is between 6.8 and 6.9.  You will use this knowledge as well as the additional notes provided:

PROVIDED NOTES: {provided_notes}

to provide a thoughtful reply to the grower with advice on what the pH value means and what to do about it if the value is outside the ideal range."""

Do RAG using the prompt.
Write the analysis to a file.
else:
    print("Wonderful! The pH is within the sweet spot.")