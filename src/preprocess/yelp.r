# reviews with tips 27555
# reviews 2225213
# tips 591864

df <- data.frame(
      variable = c("reviews with tips", "reviews without tips"),
      value = c(0.01238308, 0.9876169)
)
ggplot(df, aes(x = "", y = value, fill = variable)) +
      geom_bar(width = 1, stat = "identity") +
      scale_fill_manual(values = c("blue", "green")) +
      coord_polar("y", start = pi / 3) +
      labs(title = "percentile for review with tips in reviews")

df <- data.frame(
      variable = c("tips in review", "tips not in review"),
      value = c(0.0465563, 0.9534437)
)
ggplot(df, aes(x = "", y = value, fill = variable)) +
      geom_bar(width = 1, stat = "identity") +
      scale_fill_manual(values = c("blue", "green")) +
      coord_polar("y", start = pi / 3) +
      labs(title = "percentile for tips with reviews in tips")

dat <- data.frame(
      name = factor(c("avg_len for tips in review","avg_len for tips not in review"), levels=c("avg_len for tips in review","avg_len for tips not in review")),
      len = c(13.1, 10.7)
)
ggplot(data=dat, aes(x=name, y=len, fill=name)) +
      geom_bar(colour="black", stat="identity") + xlab("two kinds of tips") + ylab("length of tips") +
      guides(fill=FALSE)
