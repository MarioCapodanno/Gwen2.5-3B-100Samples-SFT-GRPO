# Auto-generated from 1_NLP_EDA.ipynb. Do not edit by hand.
# Original code cells flattened into a single main() for reproducibility.

def main():
    # %%capture
    # !pip install -U datasets huggingface_hub fsspec

    # %%capture
    # !pip install bertopic
    # %%capture
    # !pip install -q python-terrier==0.11.0
    # basics
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # utils
    import re
    from collections import Counter
    from tqdm import tqdm
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer
    from wordcloud import WordCloud


    # dataset
    from datasets import load_dataset, DatasetDict, concatenate_datasets

    # clustering
    from bertopic import BERTopic

    # indexing
    import pyterrier as pt
    if not pt.started():
      pt.init()
    # for reproducivity
    np.random.seed(42)

    ocr_ds_split_0 = load_dataset("nvidia/OpenCodeReasoning", "split_0")
    print(ocr_ds_split_0)

    ocr_ds_split_1 = load_dataset("nvidia/OpenCodeReasoning", "split_1")
    print(ocr_ds_split_1)

    # To complete split 1 "input" column
    datasets = {
        "taco": load_dataset("BAAI/TACO", trust_remote_code=True),
        "apps": load_dataset("codeparrot/apps", trust_remote_code=True)
    }
    # Add input to split 1
    def update_input(item):
        if item["dataset"] in ["taco", "apps"] and item["input"] == "-":
            item["input"] = datasets[item["dataset"]][item["split"]][int(item["index"])]["question"]
        return item

    ocr_ds_split_1["split_1"] = ocr_ds_split_1["split_1"].map(update_input)
    # Merge splits
    dataset = concatenate_datasets([ocr_ds_split_0["split_0"], ocr_ds_split_1["split_1"]])

    del datasets
    del ocr_ds_split_0
    del ocr_ds_split_1
    print(dataset)
    # Take smaller sample to compute data analysis
    sample_dataset = dataset.shuffle(seed=42).select(range(10000))
    df = pd.DataFrame(sample_dataset)

    df.head()
    df.head()
    # Standardize difficulties
    valid_difficulties = ["EASY", "MEDIUM", "HARD", "VERY_HARD"]
    def categorize_difficulty(d):
        if d not in [str(i) for i in range(1, 12+1)]:
          return d
        d = int(d)

        if d <= 3:
            return "EASY"
        elif d <= 7:
            return "MEDIUM"
        elif d <= 10:
            return "HARD"
        else:
            return "VERY_HARD"

    df["difficulty"] = df["difficulty"].apply(categorize_difficulty)


    mask_valid = df["difficulty"].isin(valid_difficulties)
    num_valid = mask_valid.sum()
    num_total = len(df)
    num_unknown = num_total - num_valid

    percent_valid = 100 * num_valid / num_total
    percent_invalid = 100 * num_unknown / num_total

    print(f"Examples with valid difficulty: {num_valid} ({percent_valid}%)")
    print(f"Examples with unknown difficulty: {num_unknown} ({percent_invalid}%)")

    difficulty_counts = Counter(df[mask_valid]["difficulty"])
    total = sum(difficulty_counts.values())

    percentages = {k: round(100 * v / total, 2) for k, v in difficulty_counts.items()}


    labels = sorted(list(percentages.keys()), key=lambda x: valid_difficulties.index(x))

    values = [percentages[k] for k in labels]

    colors = {'EASY': 'blue', 'MEDIUM': 'orange', 'HARD': 'green', 'VERY_HARD': 'red'}
    bar_colors = [colors[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 2))
    start = 0
    for value, color, label in zip(values, bar_colors, labels):
        ax.barh(0, value, left=start, color=color, edgecolor='black')
        ax.text(start + value / 2, 0, label.upper(), va='center', ha='center', color='white', fontsize=10, weight='bold')
        ax.text(start + value / 2, -0.1, f"({value:.1f}%)", va='center', ha='center', color='white', fontsize=9, weight='bold')

        start += value

    ax.set_xlim(0, 100)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # plot them as rectangle
    input_word_len = df['input'].apply(lambda x: len(x.split()))
    output_word_len = df['output'].apply(lambda x: len(x.split()))

    df['output_word_len'] = output_word_len

    print(f"Average Input Length: {input_word_len.mean():.2f}")
    print(f"Average Output Length: {output_word_len.mean():.2f}")

    plt.figure(figsize=(12, 6))

    sns.histplot(input_word_len, bins=50, kde=True, color='skyblue', stat="proportion")
    plt.title('Distribution of Input Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.title('Distribution of Output Lengths')
    sns.histplot(output_word_len, bins=50, kde=True, color='salmon', stat="proportion")
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(12, 6))
    for difficulty in ["EASY", "MEDIUM", "HARD", "VERY_HARD"]:
        difficulty_df = df[df['difficulty'] == difficulty]
        difficulty_output_word_len = difficulty_df['output_word_len']
        sns.histplot(difficulty_output_word_len, bins=50, kde=True, stat="proportion", alpha=0.2)

    plt.title('Distribution of Output Lengths based on difficulty')
    plt.legend(["EASY", "MEDIUM", "HARD", "VERY_HARD"])
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')

    plt.show()

    # Free ram
    del input_word_len
    del output_word_len
    input_words = df['input'].str.split().explode()
    output_words = df['output'].str.split().explode()



    def vocab_per_doc(df, col):
        return df[col].apply(lambda x: len(set(x.split())))

    input_vocab_per_doc = vocab_per_doc(df, 'input')
    output_vocab_per_doc = vocab_per_doc(df, 'output')

    print(f"\nAverage Unique Words per Problem Statement: {input_vocab_per_doc.mean():.2f}")
    print(f"Average Unique Words per Solution: {output_vocab_per_doc.mean():.2f}")

    plt.figure(figsize=(12, 6))

    sns.histplot(input_vocab_per_doc, bins=50, kde=True, color='skyblue', stat="proportion")
    plt.title('Distribution of Unique Input Words')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()
    plt.figure(figsize=(12, 6))
    sns.histplot(output_vocab_per_doc, bins=50, kde=True, color='salmon', stat="proportion")
    plt.title('Distribution of Unique Output Words')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()

    del input_vocab_per_doc
    del output_vocab_per_doc

    plt.figure(figsize=(12, 6))
    for difficulty in ["EASY", "MEDIUM", "HARD", "VERY_HARD"]:
        difficulty_df = df[df['difficulty'] == difficulty]
        difficulty_output_vocab_per_doc = vocab_per_doc(difficulty_df, 'output')
        sns.histplot(difficulty_output_vocab_per_doc, bins=50, kde=True, stat="proportion", alpha=0.2, label=difficulty)

    plt.title('Distribution of Unique Words based on difficulty')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()




    # To free ram
    del input_words
    del output_words
    think_pattern = r"<think>(.*?)</think>"
    code_pattern = r'```python(.*?)```'

    think_lengths = []
    code_lengths = []

    for output in df['output']:
        think_match = re.search(think_pattern, output, re.DOTALL)
        think_length = len(think_match.group(1).split()) if think_match else 0
        think_lengths.append(think_length)

        code_match = re.search(code_pattern, output, re.DOTALL)
        code_length = len(code_match.group(1).split()) if code_match else 0
        code_lengths.append(code_length)


    code_ratio = [100*c / (c+t) if t > 0 else float('inf') for c, t in zip(code_lengths, think_lengths)]

    code_ratio_avg = np.mean(code_ratio)
    think_ratio_avg = 100 - code_ratio_avg

    print(f"Average Code Ratio: {code_ratio_avg:.2f}%")


    fig, axs = plt.subplots(1, 2, figsize=(24, 6), gridspec_kw={'width_ratios': [9, 1]})


    sns.histplot(code_ratio, bins=50, kde=True, color='salmon', ax=axs[0])
    axs[0].set_title('Distribution of Code Ratio')
    axs[0].set_xlabel('Code Ratio (%)')
    axs[0].set_ylabel('Frequency')


    axs[1].bar(0, code_ratio_avg, color='coral', edgecolor='black', width=2)
    axs[1].bar(0, think_ratio_avg, bottom=code_ratio_avg, color='skyblue', edgecolor='black', width=2)

    axs[1].text(0, 50, 'Reasoning\nPart', va='center', ha='center', color='white', fontsize=12)

    axs[1].set_ylim(0, 100)
    axs[1].axis('off')

    plt.show()


    # help ram
    del think_lengths
    del code_lengths
    del code_ratio

    imports = df["solution"].dropna().apply(lambda x: re.findall(r"import\s+\w+", x))
    import_freq = Counter([imp.split()[-1] for sub in imports for imp in sub])
    top_imports = dict(import_freq)

    # Generate word cloud
    wc = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate_from_frequencies(top_imports)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Top Imports in solutions")
    plt.show()

    # Restructure for pyterrier format
    index_df = pd.DataFrame()

    index_df["docno"] = df["id"]
    index_df["text"] = (
        "Input:\n" + df["input"] +
        "\n\nOutput:\n" + df["output"] +
        "\n\nDifficulty:\n" + df["difficulty"].astype(str)
    )
    indexer = pt.DFIndexer("./index_code", overwrite=True)
    index_ref = indexer.index(index_df["text"], index_df["docno"])
    index_ref.toString()
    index = pt.IndexFactory.of(index_ref)
    print(index.getCollectionStatistics().toString())
    def find_problem(query):

      br = pt.terrier.Retriever(index, wmodel="TF_IDF")
      docno =  br.search(query).iloc[0]["docno"]

      return df[df["id"] == docno]["input"].iloc[0], df[df["id"] == docno]["solution"].iloc[0], df[df["id"] == docno]["difficulty"].iloc[0]
    # @title Find the problem you want!
    query = "hard dynamic programming problem" # @param{type:"string"}

    problem, sol, difficulty = find_problem(query)

    print("Problem:")
    print(problem)
    print("\nSolution:")
    print(sol)
    print("\nDifficulty:")
    print(difficulty)

    # To induce topics (doesn't actually create them)
    seed_topic_list = [
        ["array", "subarray", "index", "element", "list"],                     # Array
        ["string", "substring", "character", "palindrome", "anagram"],         # String
        ["hash", "hashmap", "dictionary", "key", "value"],                     # Hash Table
        ["dp", "dynamic programming", "state", "transition", "memoization"],   # Dynamic Programming
        ["math", "gcd", "lcm", "prime", "factorial", "modulo"],                 # Math
        ["sort", "sorted", "ascending", "descending", "merge sort", "quick sort"], # Sorting
        ["greedy", "optimal", "choice", "maximize", "minimize"],                # Greedy
        ["search", "linear search", "lookup", "find", "explore"],               # Search
        ["binary search", "midpoint", "sorted array", "left", "right"],         # Binary Search
        ["database", "query", "sql", "select", "table"],                        # Database
        ["matrix", "2D array", "row", "column", "grid"],                        # Matrix
        ["tree", "binary tree", "node", "root", "leaf"],                        # Tree
        ["heap", "priority queue", "min heap", "max heap"],                     # Heap
        ["simulation", "simulate", "state change", "process"],                 # Simulation
        ["stack", "push", "pop", "last in first out", "lifo"],                  # Stack
        ["counting", "count", "frequency", "occurrence", "combinations"],       # Counting
        ["graph", "node", "edge", "adjacency", "undirected", "directed"],        # Graph
        ["design", "class", "object", "system", "architecture"],                # Design
        ["trie", "prefix tree", "insert", "search word", "starts with"],         # Trie
        ["combinatorics", "combination", "permutation", "arrangement", "selection"], # Combinatorics
        ["bitmask", "bitwise", "mask", "binary", "or", "and"],                  # Bitmask
        ["queue", "enqueue", "dequeue", "first in first out", "fifo"],           # Queue
        ["recursion", "recursive", "base case", "call stack"],                  # Recursion
        ["divide and conquer", "split", "merge", "divide", "combine"],           # Divide and Conquer
        ["shortest path", "dijkstra", "bellman-ford", "pathfinding", "graph"],   # Shortest Path
        ["iterator", "iteration", "next element", "collection", "traversal"],    # Iterator
        ["probability", "statistics", "expected value", "variance", "distribution"], # Probability and Statistics
        ["shell", "bash", "command line", "script", "terminal"]                 # Shell
    ]

    # To avoid stopwords
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Download at the following link https://polimi365-my.sharepoint.com/:f:/g/personal/10804856_polimi_it/Erht2rCUcwtGlD2XIX3EF3MBfh9MVCjRwz52HPjKnqL2SA?e=oUk45e
    topic_model = BERTopic.load("topic_model")

    # # Generate topics (takes time)
    # topic_model = BERTopic(vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model, seed_topic_list=seed_topic_list, nr_topics=15)
    # topics, probs = topic_model.fit_transform(sample_dataset["output"])

    # # low memory option
    # vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=10)
    # topic_model = BERTopic(vectorizer_model=vectorizer_model, low_memory=True)
    # topics, probs = topic_model.fit_transform(dataset["output"])
    # Get topic information
    topic_info = topic_model.get_topic_info()
    topic_info
    def display_topic(topic_model, index):
        data = topic_model.get_topic(index)
        custom_label = topic_model.custom_labels_[index+1]


        # Create a dictionary from the list of lists
        word_freq = dict(data)

        wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        min_font_size = 10).generate_from_frequencies(word_freq)

        plt.figure(figsize = (2, 2), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.title(f"{custom_label} topic representative words")
        plt.show()

    # Examples of topics
    display_topic(topic_model, 2)
    display_topic(topic_model, 5)
    display_topic(topic_model, 10)

    topic_model.visualize_barchart(custom_labels=True)
    topic_model.visualize_topics()


if __name__ == "__main__":
    main()
