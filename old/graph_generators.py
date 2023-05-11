import matplotlib.pyplot as plt

def generate_val_train_loss_graphs(train_losses, val_losses, targetfile):    
    # Define the x-axis values
    x = range(len(train_losses))

    # Plot the training losses
    plt.plot(x, train_losses, label='Training loss')

    # Plot the validation losses
    plt.plot(x, val_losses, label='Validation loss')

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    
#     ##### DEBUGPRINT
#     print("DEBUGPRINT !!! generate_val_train_loss_graphs train_losses")
#     print(train_losses)
#     print("DEBUGPRINT !!! generate_val_train_loss_graphs val_losses")
#     print(val_losses)
    

    # Display the plot
    plt.savefig(targetfile)
