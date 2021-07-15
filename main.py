import nltk
import configparser
import random
import time

INTENTS_KEY = ["Name", "Love"]
RESPONSES_KEY = [intent_key + "_Response" for intent_key in INTENTS_KEY]
UNRECOGNIZABLE_RESPONSE = "I'm sorry doctor, but I don't understand what you're talking about"


def read_chat_file(filename):
    chats = configparser.ConfigParser()
    chats.read(filename)

    return chats


def store_chats(chats):
    vocabularies = dict()
    responses = dict()
    for intent_key, response_key in zip(INTENTS_KEY, RESPONSES_KEY):
        words_used = set(word for sent in chats[intent_key].values()
                              for word in nltk.word_tokenize(sent))
        vocabularies[intent_key] = words_used
        responses[intent_key] = [sent for sent in chats[response_key].values()]

    return vocabularies, responses

def count_probs(user_words, intent_words):
    similar_words = user_words.intersection(intent_words)
    probs = len(similar_words) / len(user_words)
    return probs


def chat_probs(user_input, vocab):
    user_words = nltk.word_tokenize(user_input.lower())
    user_words = set(user_words)
    scores = dict()

    for intent_key in vocab:
        scores[intent_key] = count_probs(user_words, vocab[intent_key])

    return scores


def unknown_check(value):
    if value == 0:
        return True

    return False


def response_matching(scores, responses):
    matched_intention = max(scores, key=scores.get)
    max_score = scores[matched_intention]

    if unknown_check(max_score):
        return UNRECOGNIZABLE_RESPONSE

    matched_response = responses[matched_intention]
    random_selections = random.randint(0, len(matched_response) - 1)

    return matched_response[random_selections]


def loading_chat():
    print("Texting", end=" ")
    for dot in range(3):
        print(".", end="")
        time.sleep(0.5)

    else:
        print(".", end="\r", flush=True)


def main():
    # Load and read chats
    chats = read_chat_file("W_chatlog.ini")
    vocab, responses = store_chats(chats)

    # Conversation loop
    while True:
        user_input = str(input())
        scores = chat_probs(user_input, vocab)
        matches = response_matching(scores, responses)

        loading_chat()

        print(matches)



if __name__ == '__main__':
    main()
