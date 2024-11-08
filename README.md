## The virtual env for the Backend RAG

pip install virtualenv

virtualenv myenv

.\myenv\Scripts\activate

## Running the server

uvicorn main:app --reload