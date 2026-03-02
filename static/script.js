const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const resultDiv = document.getElementById("result");
const loader = document.getElementById("loader");

// Preview image before sending
imageInput.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewImage.style.display = "block";
        };
        reader.readAsDataURL(file);
    }
});

async function predictImage() {
    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    resultDiv.innerHTML = "";
    loader.style.display = "block";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loader.style.display = "none";

        if (data.error) {
            resultDiv.innerHTML = "Error: " + data.error;
            return;
        }

        const className = data.class;
        const confidence = data.confidence;

        resultDiv.innerHTML = `
            Prediction: <span class="${className}">
                ${className.toUpperCase()}
            </span><br>
            Confidence: ${confidence}%
        `;

    } catch (error) {
        loader.style.display = "none";
        resultDiv.innerHTML = "Server error!";
    }
}