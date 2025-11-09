let annotations = []
const reviewOrderMap = new Map()
const dataset = []
let currentReviewIndex = 0
let importedReviews = []
let totalReviews = 0

// Initialize the application
document.addEventListener("DOMContentLoaded", () => {
  updateProgressBar()
})

function addAnnotations() {
  const entity = document.querySelector('input[name="entity"]:checked')?.value
  const attr = document.querySelector('input[name="attribute"]:checked')?.value
  const sentiment = document.querySelector('input[name="sentiment"]:checked')?.value

  if (!entity || !attr || !sentiment) {
    showNotification("Please select all options (Entity, Attribute, and Sentiment)")
    return
  }

  const annotation = {
    aspect: `${entity}#${attr}`,
    sentiment,
  }

  annotations.push(annotation)
  displayAnnotations(annotation)

  // Clear selections after adding
  clearSelections()
}

function clearSelections() {
  // Uncheck all radio buttons
  document.querySelectorAll('input[type="radio"]:checked').forEach((radio) => {
    radio.checked = false
  })
}

function displayAnnotations(annotation) {
  const li = document.createElement("li")
  li.className = `annotation-item ${annotation.sentiment}`

  const text = document.createElement("span")
  text.innerText = `${annotation.aspect}, ${annotation.sentiment}`

  const deleteBtn = document.createElement("span")
  deleteBtn.className = "delete-btn"
  deleteBtn.innerHTML = "×"
  deleteBtn.onclick = () => {
    const index = annotations.indexOf(annotation)
    if (index > -1) {
      annotations.splice(index, 1)
      li.remove()
    }
  }

  li.appendChild(text)
  li.appendChild(deleteBtn)

  document.getElementById("annotations-list").appendChild(li)
}

function clearAnnotations() {
  annotations = []
  document.getElementById("annotations-list").innerHTML = ""
}

function saveReview() {
  const reviewText = document.getElementById("review-input").value.trim()
  if (!reviewText) {
    showNotification("Enter the review please!")
    return
  }
  if (annotations.length === 0) {
    showNotification("Please add at least one annotation!")
    return
  }

  // Set unique number as index for a review Text
  let orderNumber = 1
  while (Array.from(reviewOrderMap.values()).includes(orderNumber)) {
    orderNumber++
  }
  reviewOrderMap.set(reviewText, orderNumber)

  // Add to dataset
  dataset.push({
    review: reviewText,
    annotations: [...annotations],
    orderNumber: orderNumber,
  })

  // Sort dataset by orderNumber
  dataset.sort((a, b) => a.orderNumber - b.orderNumber)
  updateDatasetDisplay()
  clearAnnotations()
  nextReview()

  // Show success notification
  showNotification("Review saved successfully!", "success")

  // Update progress
  updateProgressBar()
}

function updateProgressBar() {
  const progressFill = document.getElementById("progress-fill")
  const progressText = document.getElementById("progress-text")

  if (totalReviews === 0) {
    progressFill.style.width = "0%"
    progressText.textContent = "0/0 Reviews"
    return
  }

  const progress = (Math.min(currentReviewIndex, totalReviews) / totalReviews) * 100
  progressFill.style.width = `${progress}%`
  progressText.textContent = `${Math.min(currentReviewIndex, totalReviews)}/${totalReviews} Reviews`
}

function updateDatasetDisplay() {
  const dataset_output = document.getElementById("annotated-dataset")
  dataset_output.innerHTML = ""

  dataset.forEach((item, index) => {
    const reviewDiv = document.createElement("div")
    reviewDiv.className = "review-item"

    // Create review content
    const reviewContent = document.createElement("div")
    reviewContent.className = "review-content"
    reviewContent.innerHTML = `<strong>Review #${item.orderNumber}:</strong> ${item.review}`

    // Create annotations display
    const annotationsDiv = document.createElement("div")
    annotationsDiv.className = "review-annotations"

    // Format annotations with color coding
    const annotationsHtml = item.annotations
      .map((ann) => {
        let colorClass = ""
        if (ann.sentiment === "positive") colorClass = "color: var(--positive-color)"
        else if (ann.sentiment === "negative") colorClass = "color: var(--negative-color)"
        else colorClass = "color: var(--neutral-color)"

        return `<span style="${colorClass}">{${ann.aspect}, ${ann.sentiment}}</span>`
      })
      .join(", ")

    annotationsDiv.innerHTML = `<strong>Annotations:</strong> ${annotationsHtml}`

    // Create delete checkbox
    const deleteCheckboxDiv = document.createElement("div")
    deleteCheckboxDiv.className = "delete-checkbox"

    const checkbox = document.createElement("input")
    checkbox.type = "checkbox"
    checkbox.id = `review-${index}`
    checkbox.style.display = "none"

    const checkboxLabel = document.createElement("label")
    checkboxLabel.htmlFor = `review-${index}`
    checkboxLabel.innerHTML = "<span>Delete</span>"

    checkbox.onchange = function () {
      if (this.checked) {
        deleteReview(index)
      }
    }

    deleteCheckboxDiv.appendChild(checkbox)
    deleteCheckboxDiv.appendChild(checkboxLabel)

    // Append all elements
    reviewDiv.appendChild(reviewContent)
    reviewDiv.appendChild(annotationsDiv)
    reviewDiv.appendChild(deleteCheckboxDiv)

    dataset_output.appendChild(reviewDiv)
  })
}

function deleteReview(index) {
  const review = dataset[index]
  dataset.splice(index, 1)
  reviewOrderMap.delete(review.review)
  updateDatasetDisplay()
  showNotification("Review deleted successfully!", "success")
}

function downloadAnnotations() {
  if (dataset.length === 0) {
    showNotification("No data to export!")
    return
  }

  // Get rid of orderNumber
  const datasetWithoutOrderNumber = dataset.map(({ orderNumber, ...rest }) => rest)
  const dataStr =
    "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(datasetWithoutOrderNumber, null, 2))
  const dlAnchor = document.createElement("a")
  dlAnchor.setAttribute("href", dataStr)
  dlAnchor.setAttribute("download", "annotations.json")
  dlAnchor.click()

  showNotification("Dataset downloaded successfully!", "success")
}

function importAnnotations() {
  const fileInput = document.getElementById("file-input")
  if (!fileInput.files.length) {
    showNotification("Please select a file!")
    return
  }

  const file = fileInput.files[0]
  const reader = new FileReader()
  reader.onload = (e) => {
    importedReviews = e.target.result
      .split(/(?:^|\n)#\d+/)
      .map((s) => s.trim())
      .filter((s) => s)

    currentReviewIndex = 0
    totalReviews = importedReviews.length
    updateProgressBar()
    nextReview()
    showNotification(`${importedReviews.length} reviews imported successfully!`, "success")
  }

  reader.readAsText(file)
}

function nextReview() {
  if (currentReviewIndex < importedReviews.length) {
    document.getElementById("review-input").value = importedReviews[currentReviewIndex]
    document.getElementById("review-number").textContent = `Review #${currentReviewIndex + 1}`
    clearAnnotations()
    currentReviewIndex++
    updateProgressBar()
  } else {
    showNotification("No more reviews!")
  }
}

function prevReview() {
  if (currentReviewIndex > 1) {
    currentReviewIndex -= 2
    nextReview()
  } else {
    showNotification("No previous reviews!")
  }
}

// Notification system
function showNotification(message, type = "error") {
  // Create notification element if it doesn't exist
  let notification = document.getElementById("notification")
  if (!notification) {
    notification = document.createElement("div")
    notification.id = "notification"
    notification.style.position = "fixed"
    notification.style.bottom = "20px"
    notification.style.right = "20px"
    notification.style.padding = "10px 20px"
    notification.style.borderRadius = "4px"
    notification.style.color = "white"
    notification.style.fontWeight = "bold"
    notification.style.zIndex = "1000"
    notification.style.boxShadow = "0 3px 6px rgba(0,0,0,0.16)"
    notification.style.transition = "all 0.3s ease"
    document.body.appendChild(notification)
  }

  // Set notification style based on type
  if (type === "error") {
    notification.style.backgroundColor = "var(--danger-color)"
  } else if (type === "success") {
    notification.style.backgroundColor = "var(--success-color)"
  } else if (type === "warning") {
    notification.style.backgroundColor = "var(--warning-color)"
  }

  // Set message and show notification
  notification.textContent = message
  notification.style.opacity = "1"

  // Hide notification after 3 seconds
  setTimeout(() => {
    notification.style.opacity = "0"
  }, 3000)
}

