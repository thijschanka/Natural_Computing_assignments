import numpy as np
from sklearn import metrics
from matplotlib import pyplot
import os


def convert_to_chunks(text, length=100):
    return [text[i:i + length] if i + length <= len(text) else text[i:].ljust(length, ' ') for i in
            range(0, len(text), length)]


for input_type in ["snd-cert", "snd-unm"]:
    for chunk_length in [4, 8]:
        for r in [2, 4]:
            classes = []
            true_classes = []
            test_chunks = []
            scores = []

            print(f"Type: {input_type}, n: {chunk_length}, r: {r}")

            print("Reformatting files.")

            with open(f"negative-selection/syscalls/{input_type}/{input_type}.train") as train_file:
                text = train_file.readlines()
                chunks = "\n".join(["\n".join(convert_to_chunks(e.split('\n')[0], chunk_length)) for e in text])

                if not os.path.exists(
                        f"negative-selection/syscalls/{input_type}/chunked-{input_type}-{chunk_length}-{r}.train"):
                    with open(f"negative-selection/syscalls/{input_type}/chunked-{input_type}-{chunk_length}-{r}.train",
                              "w") as output_file:
                        output_file.write(chunks)

            for i in range(1, 4):
                with open(f"negative-selection/syscalls/{input_type}/{input_type}.{i}.labels") as labels_file:
                    with open(f"negative-selection/syscalls/{input_type}/{input_type}.{i}.test") as test_file:
                        labels = labels_file.readlines()
                        test_text = test_file.readlines()

                        chunks = [(convert_to_chunks(e.split('\n')[0], chunk_length),
                                   len(convert_to_chunks(e.split('\n')[0], chunk_length))) for e in test_text]

                        for i in range(len(chunks)):
                            classes.append((labels[i].split('\n')[0], chunks[i][1]))
                            for j in range(chunks[i][1]):
                                test_chunks.append(chunks[i][0][j])

            if not os.path.exists(
                    f"negative-selection/syscalls/{input_type}/chunked-{input_type}-{chunk_length}-{r}.test"):
                with open(f"negative-selection/syscalls/{input_type}/chunked-{input_type}-{chunk_length}-{r}.test",
                          "w") as output_file:
                    output_file.write("\n".join(test_chunks))

            if not os.path.exists(
                    f"negative-selection/syscalls/{input_type}/chunked-{input_type}-{chunk_length}-{r}.labels"):
                with open(f"negative-selection/syscalls/{input_type}/chunked-{input_type}-{chunk_length}-{r}.labels",
                          "w") as output_file:
                    output_file.write("\n".join(classes))

            print("Evaluating.")

            if not os.path.exists(f"output-{input_type}-{chunk_length}-{r}.txt"):
                os.system(f"java -jar negative-selection/negsel2.jar \
                        -alphabet file://negative-selection/syscalls/{input_type}/{input_type}.alpha \
                        -self negative-selection/syscalls/{input_type}/chunked-{input_type}-{chunk_length}-{r}.train \
                        -n {chunk_length} -r {r} -c -l \
                        < negative-selection/syscalls/{input_type}/chunked-{input_type}-{chunk_length}-{r}.test \
                        > output-{input_type}-{chunk_length}-{r}.txt")

            print("Plotting.")

            with open(f"output-{input_type}-{chunk_length}-{r}.txt") as output_file:
                output = output_file.readlines()

                i = 0
                true_i = 0

                while i < len(output):
                    true_classes.append(classes[true_i][0])
                    composition_scores = []
                    for j in range(classes[true_i][1]):
                        if not output[i].isspace():
                            composition_scores.append(float(output[i]))
                            i += 1

                    scores.append(np.nanmean(composition_scores))

                    true_i += 1

            classes = np.array(true_classes, dtype=np.int)
            scores = np.array(scores)
            not_nan = np.isnan(scores) == False

            fpr, tpr, thresholds = metrics.roc_curve(classes[not_nan], scores[not_nan], pos_label=1)

            auc = metrics.roc_auc_score(classes[not_nan], scores[not_nan])

            pyplot.plot(fpr, tpr, linestyle='--', label="Roc curve (AUC = " + str(auc) + ")")
            pyplot.xlabel("Specificity")
            pyplot.ylabel("Sensitivity")
            pyplot.legend()
            pyplot.savefig(f"ex2-{input_type}-{chunk_length}-{r}.pdf")
            pyplot.show()

            print()
