# Rest API implementation for TuneHive Music Recommendation System

The recommender system used here is the GRU4REC with Attention Mechanism. Creating by ML Capstone Team C242-PS288

 ---

## Guide to Use

Here's how to run this API:

1. Open your terminal, then open the path to the project root directory
2. If you haven't setup the virtual environment and install the dependencies, you can run ```python -m venv .venv``` to initialize virtual environment, and then run ```pip install -r requirements.txt``` to install all the dependencies required.
3. Type ```fastapi dev main.py ``` or ```uvicorn main:app --reload``` to run the application. For the deployment purpose, use the uvicorn one
4. For the documentation of the APIs, you can check the "/docs" endpoint of application when running (only in dev mode)

 ---

## Other Tips

To update the dependencies conveniently, you can use ```pipreqs``` to update the dependencies. install the pipreqs on your virtual environment by running ```pip install pipreqs```. After that, you can update the `requirements.txt` dependencies by running `pipreqs . --ignore .venv` on the terminal at the root path of the project