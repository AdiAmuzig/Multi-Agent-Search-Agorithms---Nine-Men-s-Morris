import matplotlib.pyplot as plt

# Initial depth 3:
#   Light Depth == 3, Heavy Depth == 3:
#       Light Wins: 0    Heavy Wins: 2 // all wins
#   Light Depth == 4, Heavy Depth == 3:
#       Light Wins: 1    Heavy Wins: 1 // 1 win 1 lose
#   Light Depth == 5, Heavy Depth == 3:
#       Light Wins: 0    Heavy Wins: 2

# Initial depth 2:
#   Light Depth == 2, Heavy Depth == 2:
#       Light Wins: 0    Heavy Wins: 2
#   Light Depth == 3, Heavy Depth == 2:
#       Light Wins: 0    Heavy Wins: 2
#   Light Depth == 4, Heavy Depth == 2:
#       Light Wins: 0.5    Heavy Wins: 1.5 // tie

# Notice that Heavy could win in all of thenon the grounds of move_time if small enough.
# especially in the initial moves

if __name__ == "__main__":
    x_depth_diff = [0, 1, 2]
    y_depth2 = [2/2, 2/2, 1.5/2]
    y_depth3 = [2/2, 1/2, 2/2]

    plt.plot(x_depth_diff, y_depth2, 'r^--')
    plt.title('Initial Depth = 2')
    plt.xlabel('Difference In Depth')
    plt.ylabel('HeavyPlayer Victories over LightPlayer')
    plt.grid(color='lightgray', linestyle='--')
    plt.ylim([-0.1, 1.1])
    plt.show()

    plt.plot(x_depth_diff, y_depth3, 'bo:')
    plt.title('Initial Depth = 3')
    plt.xlabel('Difference In Depth')
    plt.ylabel('HeavyPlayer Victories over LightPlayer')
    plt.grid(color='lightgray', linestyle='--')
    plt.ylim([-0.1, 1.1])
    plt.show()
