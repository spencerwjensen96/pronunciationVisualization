# Setting up a Virtual Environment with venv

To create a virtual environment using venv, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the root directory of this `streamlit` project.
3. Run the following command to create a virtual environment:

    ```bash
    python3 -m venv myenv
    ```

    This will create a new directory named `myenv` that contains the virtual environment.

4. Activate the virtual environment by running the appropriate command for your operating system:

    - For macOS and Linux:

      ```bash
      source myenv/bin/activate
      ```

    - For Windows:

      ```bash
      myenv\\Scripts\\activate
      ```

    You will see `(myenv)` prefix in your terminal, indicating that the virtual environment is active.

5. Confirm that the virutal environment is running:

    ```bash
    which python
    ```

# Installing PyPackages using a requirements.txt file

To install PyPackages using the requirements.txt file, follow these steps:

1. Make sure your virtual environment is activated.
2. Run the following command to install the python packages:

    ```bash
    pip install -r requirements.txt
    ```

    This will install all the packages listed in the `requirements.txt` file.

# Starting a Streamlit App

To start a Streamlit app, follow these steps:

1. Make sure your virtual environment is activated and all the required PyPackages are installed.
2. Run the following command to start the Streamlit app:

    ```bash
    streamlit run app.py
    ```

    This will start the Streamlit app, and you will see a local URL where you can view and interact with your app.

That's it!