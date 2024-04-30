function calculateAge() {
    // Get user input
    const birthYear = document.getElementById("birthYear").value;
    const birthMonth = document.getElementById("birthMonth").value - 1; // Months are 0-indexed
    const birthDate = document.getElementById("birthDate").value;

    // Get today's date
    const today = new Date();
    const currentYear = today.getFullYear();
    const currentMonth = today.getMonth();
    const currentDate = today.getDate();

    // Calculate age in years
    let ageInYears = currentYear - birthYear;

    // Check if birthday has passed in the current year
    if (currentMonth < birthMonth || (currentMonth === birthMonth && currentDate < birthDate)) {
        ageInYears--;
    }

    // Display the result (corrected logic)
    const result = document.getElementById("result");
    result.textContent = ageInYears > 0 ? `Your age is: ${ageInYears} years old.` : "You haven't had your birthday yet this year.";
}