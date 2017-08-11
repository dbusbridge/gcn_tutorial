library(igraph)
library(tidyverse)

g <- make_graph("Zachary")
oc <- cluster_optimal(g)

colours <- c("red", "green", "blue", "cyan")

V(g)$membership <- oc$membership
V(g)$color <- map(V(g)$membership, function(x) colours[x]) %>% as.character

g %>% igraph::write.graph("karate.graphml", format = "graphml")

