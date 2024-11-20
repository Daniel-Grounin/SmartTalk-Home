import ollama
from gtts import gTTS
import os
import re


def clean_hebrew_text(text):
    # Keep only Hebrew characters and spaces
    cleaned_text = re.sub(r'[^\u0590-\u05FF\s]', '', text)  # Unicode range for Hebrew
    return cleaned_text.strip()


def chat_with_tinydolphin():
    print("Welcome to TinyDolphin Chat in Hebrew! Type 'exit' or 'quit' to end the conversation.\n")

    # Open a file to log all conversations
    with open("tinydolphin_responses_hebrew.txt", "a", encoding="utf-8") as log_file:
        while True:
            user_input = input("You: ")

            # Exit the loop if the user types 'exit' or 'quit'
            if user_input.lower() in ['exit', 'quit']:
                print("להתראות!")  # "Goodbye!" in Hebrew
                break

            try:
                stream = ollama.chat(
                    model="tinydolphin",
                    messages=[{'role': 'user', 'content': f"{user_input}"}],
                    stream=True
                )

                # Collect the AI's response
                response = ""
                print("TinyDolphin:", end=" ", flush=True)
                for chunk in stream:
                    response += chunk['message']['content']
                    print(chunk['message']['content'], end="", flush=True)
                print("\n")  # Newline for clean formatting

                # Log the conversation to a file
                log_file.write(f"You: {user_input}\n")
                log_file.write(f"TinyDolphin: {response}\n\n")

                # Clean the response and check for Hebrew text
                cleaned_response = clean_hebrew_text(response)
                # if cleaned_response:
                #     # Convert the cleaned response to speech in Hebrew
                #     tts = gTTS(text=cleaned_response, lang='he')
                #     tts.save("response_hebrew.mp3")
                #     os.system("start response_hebrew.mp3")  # For Windows; use "open" on macOS
                # else:
                #     print("Response does not contain enough Hebrew text for speech synthesis.\n")

            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    chat_with_tinydolphin()
