# run_chat.py
from src.chat_agent import ChatAgent

def main():
    agent = ChatAgent()
    print("\n Conversational ML (gpu_tools). Type 'quit' to exit.")
    print("Try:\n  • set the dataset to data/Titanic-Dataset-1.csv or data/Titanic-Dataset.csv target=Survived"
          "\n  • describe the data"
          "\n  • preview 5"
          "\n  • train classification"
          "\n  • hyperparameter optimization (HPO)\n")
    while True:
        try:
            user = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user.lower() in {"quit", "exit"}:
            print("Bye!")
            break
        reply = agent.chat(user)
        print(f"\nAgent > {reply}\n")

if __name__ == "__main__":
    main()
