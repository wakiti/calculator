document.addEventListener("DOMContentLoaded", function() {
    const conversionTypeSelect = document.getElementById("conversionType");
    const temperatureInput = document.getElementById("temperature");
    const resultInput = document.getElementById("result");
    const convertButton = document.getElementById("convert");

    convertButton.addEventListener("click", function() {
        const conversionType = conversionTypeSelect.value;
        const temperature = parseFloat(temperatureInput.value);
        let result;
        let unit;

        if (conversionType === "celsiusToFahrenheit") {

            result = (temperature * 9 / 5) + 32;
            unit = "°F";
        } else if (conversionType === "fahrenheitToCelsius") {

            result = (temperature - 32) * 5 / 9;
            unit = "°C";
        }


        resultInput.value = result.toFixed(2) + " " + unit;
    });
});