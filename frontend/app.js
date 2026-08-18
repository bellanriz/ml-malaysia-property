const API_URL = "http://localhost:8000";

// Format number with commas: 1250000 → "1,250,000"
function formatNumber(num) {
    return Math.round(num).toLocaleString();
}

// Load dropdown options from the API
async function loadOptions() {
    try {
        const response = await fetch(API_URL + "/options");
        const data = await response.json();

        populateSelect("location", data.locations);
        populateSelect("property-type", data.property_types);
        populateSelect("furnishing", data.furnishings);
    } catch (error) {
        console.error("Failed to load options:", error);
        alert("Cannot connect to API. Make sure backend is running (python backend/api.py)");
    }
}

// Fill a <select> element with options
function populateSelect(id, options) {
    const select = document.getElementById(id);
    select.innerHTML = "";
    options.forEach(function(option) {
        const el = document.createElement("option");
        el.value = option;
        el.textContent = option;
        select.appendChild(el);
    });
}

// Send property details to API and display prediction
async function predictPrice() {
    const location = document.getElementById("location").value;
    const propertyType = document.getElementById("property-type").value;
    const furnishing = document.getElementById("furnishing").value;
    const size = parseInt(document.getElementById("size").value);
    const rooms = parseInt(document.getElementById("rooms").value);
    const bathrooms = parseInt(document.getElementById("bathrooms").value);
    const carParks = parseInt(document.getElementById("car-parks").value);

    const body = {
        location: location,
        property_type: propertyType,
        furnishing: furnishing,
        size: size,
        rooms: rooms,
        bathrooms: bathrooms,
        car_parks: carParks
    };

    try {
        const response = await fetch(API_URL + "/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Show results
        document.getElementById("predicted-price").textContent = "RM " + formatNumber(data.predicted_price);
        document.getElementById("price-sqft").textContent = "RM " + formatNumber(data.price_per_sqft);
        document.getElementById("monthly").textContent = "RM " + formatNumber(data.monthly_payment);
        document.getElementById("down-payment").textContent = "RM " + formatNumber(data.down_payment);
        document.getElementById("results").classList.remove("hidden");
    } catch (error) {
        console.error("Prediction failed:", error);
        alert("Prediction failed. Check if the API is running.");
    }
}

// Update slider labels when user moves them
function setupSliders() {
    const sliders = ["rooms", "bathrooms", "car-parks"];
    sliders.forEach(function(id) {
        const slider = document.getElementById(id);
        const label = document.getElementById(id + "-value");
        slider.addEventListener("input", function() {
            label.textContent = slider.value;
        });
    });
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", function() {
    loadOptions();
    setupSliders();
    document.getElementById("predict-btn").addEventListener("click", predictPrice);
});
