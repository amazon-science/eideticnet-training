library(glue)
library(AlgDesign)
library(jsonlite)


plot_designs <- function(label, mnist, designs) {
	mask <- mnist$train$labels == label
	images <- mnist$train$images[mask,]
	pca <- prcomp(images, retx=TRUE, center=TRUE)
	for (i in seq(from=1, to=length(designs))) {
		dev.new()
        design <- designs[[i]]
        num_points <- length(design)
		plot(
			pca$x[, 1:2],
            main=glue("Design with {num_points} points for label {label}")
		)
	    points(pca$x[design, 1:2], col="red", bg="red")
	}
}


make_design <- function(
    images, num_trials, num_components, rows=NULL
) { 
    pca <- prcomp(images, retx=TRUE, center=TRUE)
    reduced_dim_images <- pca$x[, 1:num_components]
    if (is.null(rows)) {
        # Create a new design from scratch.
        design <- AlgDesign::optFederov(
            data=reduced_dim_images, nTrials=num_trials
        )
    } else {
        # Augment the previous design.
        design <- AlgDesign::optFederov(
            data=reduced_dim_images,
            nTrials=num_trials,
            rows=rows,
            augment=TRUE
        )
    }
    design
}


make_design_for_label <- function(label, mnist, num_designs=10, num_components=2) {
    # Make designs of increasing size.
    mask <- mnist$train$labels == label
    images <- mnist$train$images[mask, ]
    num_images = dim(images)[1]
    step_size <- floor(num_images / num_designs)
    # The rows returned is a list of lists. Each list comprises row numbers
    # of that design. Design i+1 includes design i.
    rows <- list()
    i <- 1
    for (num_trials in seq(from=step_size, to=num_images, by=step_size)) {
        print(glue("{num_trials}/{num_images}"))
        if (length(rows) == 0) {
            flattened_rows <- NULL
        } else {
            flattened_rows <- rows[[i - 1]]
        }
        design <- make_design(
            images,
            num_trials=num_trials,
            num_components=num_components,
            rows=flattened_rows
        )
        rows[[i]] <- design$rows
        i <- i + 1
    }
    rows
}


make_mnist_designs <- function(mnist_path, num_components=10) {
    mnist <- dslabs::read_mnist(path=mnist_path)
    designs <- list()
    for (label in seq(0, 9)) {
        designs[[label + 1]] <- make_design_for_label(
            label, mnist, num_components=num_components
        )
    }
    names(designs) <- paste(seq(0, 9))
    output_path <- glue("mnist-designs-{num_components}-components.json")
    write_json(designs, output_path)
}
