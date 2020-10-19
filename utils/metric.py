def acc(pred_y, true_y):
    term_count = 0
    predict_count = 0
    correct_count = 0
    exact_count = len(true_y)
    exact_correct = 0
    for i in range(len(true_y)):
        term_count += len(true_y[i])
        predict_count += len(pred_y[i])
        for x in pred_y[i]:
            if x in true_y[i]:
                correct_count += 1
        if set(pred_y[i]) == set(true_y[i]):
            exact_correct += 1
    print(f"Term count: {term_count}")
    print(f"Term predict: {predict_count}")
    print(f"Term correct: {correct_count}")
    print(f"Term recall: {100 * correct_count / term_count}")
    print(f"Term precision: {100 * correct_count / predict_count}")
    print(f"Term F1: {200 * correct_count / (term_count + predict_count)}")
    print("---")
    print(f"All count: {exact_count}")
    print(f"Acc: {exact_correct / exact_count}")
    print("---")
    print(f"Estimated f1 for rule-based: {2/(1 + 0.01 * predict_count / correct_count + term_count / correct_count)}")
    return exact_correct / exact_count, 100 * correct_count / predict_count, 100 * correct_count / term_count, 200 * correct_count / (term_count + predict_count)
