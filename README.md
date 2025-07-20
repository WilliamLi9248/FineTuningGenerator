# FitTracker - iOS Fitness App

A comprehensive fitness tracking app for iOS built with SwiftUI that helps users monitor their exercise, diet, and health goals.

## Features

### 🏃‍♂️ Exercise Tracking
- Add exercises with calorie burn calculations based on user weight
- Choose from 16+ predefined exercise templates
- Create custom exercises with manual calorie input
- Track exercise duration and type
- View exercise history and weekly progress
- Visual progress charts

### 🥗 Diet Management
- Track food intake with detailed nutritional information
- Search from comprehensive food database (30+ foods)
- Browse foods by category (fruits, vegetables, proteins, etc.)
- View macro-nutrients (protein, carbs, fat, fiber)
- Meal suggestions for breakfast, lunch, dinner, and snacks
- Water intake tracking with visual indicators

### 👤 User Profile & Health Stats
- Complete user onboarding with personal information
- BMI calculation and health category assessment
- BMR (Basal Metabolic Rate) calculation
- Daily calorie needs based on activity level
- Health recommendations based on BMI
- Goal setting and progress tracking

### 📊 Dashboard & Analytics
- Daily calorie balance visualization
- Circular progress indicators
- Weekly exercise and nutrition statistics
- Recent activity overview
- Quick action buttons for common tasks

## Technical Architecture

### Models
- **User**: Complete user profile with health calculations
- **Exercise**: Exercise tracking with calorie burn algorithms
- **Food**: Comprehensive food database with nutritional data
- **FoodEntry**: Individual food consumption tracking

### Managers
- **UserManager**: User authentication and profile management
- **ExerciseManager**: Exercise tracking and statistics
- **DietManager**: Food tracking and meal suggestions

### Views
- **Dashboard**: Main overview with progress tracking
- **ExerciseView**: Exercise logging and history
- **DietView**: Food tracking and meal management
- **ProfileView**: User profile and settings

## Getting Started

### Prerequisites
- iOS 15.0+
- Xcode 13.0+
- Swift 5.5+

### Installation
1. Open the project in Xcode
2. Select your target device or simulator
3. Build and run the project

### First Launch
1. Complete the onboarding process:
   - Enter your name, height, weight, age
   - Select your gender
   - Choose your activity level
2. Set your daily calorie goal
3. Start tracking your exercises and meals

## Usage

### Adding Exercises
1. Go to the Exercise tab
2. Tap the "+" button
3. Choose from:
   - **Quick Add**: Select from predefined exercises
   - **Custom**: Create your own exercise
4. Enter duration and tap "Add Exercise"

### Tracking Food
1. Go to the Diet tab
2. Tap the "+" button
3. Select meal type (breakfast, lunch, dinner, snack)
4. Choose from:
   - **Search**: Find specific foods
   - **Browse**: Browse by category
   - **Suggestions**: Pre-made meal suggestions
5. Enter amount and tap "Add Food"

### Water Tracking
- Use the water tracker in the Diet tab
- Quick add buttons for common amounts (250ml, 500ml, 1L)
- Visual progress with water glass indicators

### Viewing Progress
- Dashboard shows daily overview
- Exercise tab shows weekly progress charts
- Profile tab displays health statistics

## Food Database

The app includes a comprehensive food database with:
- **Fruits**: Apple, Banana, Orange, Berries, etc.
- **Vegetables**: Broccoli, Spinach, Carrots, etc.
- **Proteins**: Chicken, Fish, Eggs, Tofu, etc.
- **Grains**: Rice, Quinoa, Oats, Bread, etc.
- **Dairy**: Milk, Cheese, Yogurt, etc.
- **Nuts & Seeds**: Almonds, Walnuts, etc.

Each food item includes:
- Calories per 100g
- Protein content
- Carbohydrates
- Fat content
- Fiber
- Sugar
- Sodium

## Exercise Templates

Predefined exercises include:
- Walking, Running, Cycling
- Swimming, Weightlifting, Yoga
- Dancing, Hiking, Sports
- And more with accurate calorie calculations

## Health Calculations

### BMI (Body Mass Index)
- Weight (kg) / Height (m)²
- Categories: Underweight, Normal, Overweight, Obese

### BMR (Basal Metabolic Rate)
- Uses Mifflin-St Jeor Equation
- Accounts for gender, age, weight, height

### Daily Calorie Needs
- BMR × Activity Level Multiplier
- Activity levels from sedentary to extra active

## Data Persistence

- All data is stored locally using UserDefaults
- User profile, exercises, and food entries are preserved
- No network connection required

## Future Enhancements

- Integration with HealthKit
- Social features and challenges
- More detailed analytics
- Barcode scanning for foods
- Custom meal planning
- Export data functionality

## Contributing

This is a sample fitness app demonstrating SwiftUI best practices. Feel free to extend and customize based on your needs.

## License

This project is for educational purposes and demonstrates iOS app development with SwiftUI. 