library(ggplot2)

cm_df <- data.frame(
  True      = factor(c("0","0","1","1")),
  Predicted = factor(c("0","1","0","1")),
  Value     = c(0.94, 0.06, 0.03, 0.97)   
  # Replace the values above with confusion matrix values
# from the current DeepEthogram experiment.
)

ggplot(cm_df, aes(x = Predicted, y = True, fill = Value)) +
  geom_tile() +
  geom_text(aes(label = Value),
            color    = "black",   
            size     = 9,
            fontface = "bold") +
  scale_fill_gradientn(
    colors = hcl.colors(50, "blues")
  ) +
  scale_y_discrete(limits = rev) +
  labs(title = "Confusion Matrix",
       x = "Predicted label",
       y = "True label",
       fill = "") +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid       = element_blank(),
    legend.position  = "right", 
    plot.title       = element_text(hjust = 0.5, face = "bold")
  )
