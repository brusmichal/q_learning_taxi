from Q_Learning import Q_Learning


def main():
    q_learning_1 = Q_Learning(0.1, 0.6, 0.1, 1000)
    q_learning_1.learn()
    q_learning_1.plot_learning_process()
    q_learning_1.plot_epsilons()
    q_learning_1.show_map((4, 1, 2, 0), with_training=False)
    q_learning_1.show_map((4, 1, 2, 0), with_training=True)


if __name__ == '__main__':
    main()
