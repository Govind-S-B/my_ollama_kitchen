# My Ollama Kitchen
Experimenting with a bunch of stuff in ollama , this repo is meant to record things I test out

## ollama_api_playground.py
### Why I wrote it
for experimenting with models and their parameters on the fly
lets say you have fixed a few base parameters
and now you want to play around with just a few parameters and see how they perform , you probably want this to just override the base config that you set up ... which you can do now

an example :
```
config = OllamaConfiguration(model="openhermes2.5-mistral",
                             temperature=0.4, stream=False, top_k=10, format="json")

prompt = "For the question 'A stock went down 30% over the night , how would you react to it' a user responded with 'Man, thats actually scary I cannot see that much of a swing\
    in values' . Classify this user's risk_tolerance as 'high' 'medium' or 'low' in a json format"

print(json.loads(config.generate_response(
    prompt=prompt, temperature=0).text)["response"])
```
