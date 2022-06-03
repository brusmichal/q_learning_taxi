from Q_Learning import Q_Learning


def main():
    q_learning_1 = Q_Learning(0.5, 0.6, 1, 1000)
    q_learning_1.learn()
    q_learning_1.plot_learning_process()
    q_learning_1.plot_epsilons()
    q_learning_1.evaluate(50, with_training=False)
    # q_learning_1.evaluate(50, with_training=True)
    # q_learning_1.show_map((4, 1, 2, 0), False, 50)
    # q_learning_1.show_map((4, 1, 2, 0), True, 50)


if __name__ == '__main__':
    main()
