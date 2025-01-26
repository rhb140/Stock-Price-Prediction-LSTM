package roryhb;

/**
 * Currency Converter using ExchangeRate-API
 * API Source: https://www.exchangerate-api.com/
 * Documentation: https://www.exchangerate-api.com/docs/free
 * 
 * Fetches real-time exchange rates and performs currency conversions.
 */


import java.awt.FlowLayout;
import java.net.HttpURLConnection;
import java.net.URI;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;

import org.json.JSONObject;


public class CurrencyConverter {
    //variable for exachange rate from the api
    private static Map<String, Double> exchangeRates = new HashMap<>();
    //variables for display   
    private static JTextField amountF;
    private static JLabel resultLabel;
    //variables for UI
    private static JComboBox<String> CurrencyToInput;
    private static JComboBox<String> CurrencyFromInput;


    public static void main(String[] args){
        //Create the GUI
        JFrame GUI = new JFrame("Currency Converter"); 
        GUI.setLayout(new FlowLayout()); // set flowlayout
        GUI.setSize(600, 400); // set size
        GUI.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // set window close on exit

        try{
            //call the method to get the convertions from the api
            getExchangeRates();

            //add a combobox on the GUI for users to select the input currency 
            GUI.add(new JLabel("From (Currency):"));
            CurrencyFromInput = new JComboBox<>(exchangeRates.keySet().toArray(new String[0]));
            CurrencyFromInput.setEditable(true); // Allow manual entry
            GUI.add(CurrencyFromInput);

            //add a manual input on the GUI for users to enter value to be converted
            GUI.add(new JLabel("Amount:"));
            amountF = new JTextField(10); //Create a text field for user input
            GUI.add(amountF);

            //add a combobox on the GUI for users to select the output currency 
            GUI.add(new JLabel("To (Currency):"));
            CurrencyToInput = new JComboBox<>(exchangeRates.keySet().toArray(new String[0]));
            CurrencyToInput.setEditable(true);
            GUI.add(CurrencyToInput);

            //add a button on GUI to convert currency amount when clicked
            JButton convertButton = new JButton("Convert"); // Create a button
            convertButton.addActionListener(e -> calculateConversion()); // call calculateConversion() when clicked
            GUI.add(convertButton); // Add button to GUI

            //displays the converted amount
            resultLabel = new JLabel("Converted Amount: ");
            GUI.add(resultLabel);



        } catch (Exception e) {
            System.out.println("Error occurred: " + e.getMessage());
        }
        

        // Display the frame
        GUI.setVisible(true);
    }
    //gets the extange rates from the api    
    private static void getExchangeRates() throws Exception{
        //url to api
        String url = "https://api.exchangerate-api.com/v4/latest/USD";
        HttpURLConnection APIConnection = (HttpURLConnection) new URI(url).toURL().openConnection();
        APIConnection.setRequestMethod("GET"); // get the information form api
        
        // Read the API
        Scanner apiScanner = new Scanner(APIConnection.getInputStream());
        StringBuilder response = new StringBuilder();

        while (apiScanner.hasNext()){
            response.append(apiScanner.nextLine()); // append each line of the response
        }
        apiScanner.close(); // close scanner

        //get the data from the JSON response and find the exchange rates
        JSONObject jsonResponse = new JSONObject(response.toString());
        JSONObject rates = jsonResponse.getJSONObject("rates");

        // Store extange rates in a hashmap
        for (String key: rates.keySet()) {
            exchangeRates.put(key, rates.getDouble(key));
        }
    }
    // does calulations for the convertion
    private static void calculateConversion(){
        try{
            //get user input
            double amount = Double.parseDouble(amountF.getText()); // Get amount entered by user
            String userCurrency = (String) CurrencyFromInput.getSelectedItem(); // Source currency
            String convertCurrency = (String) CurrencyToInput.getSelectedItem(); // Target currency
            
            //check if valide user input for the userCurrency and convertCurrency adn return an error message if not
            if (!exchangeRates.containsKey(userCurrency) || !exchangeRates.containsKey(convertCurrency)){
                resultLabel.setText("Invalid currency selection.");
                return;
            }

            //get the convertion rates based on the user input
            double fromCurrency = exchangeRates.get(userCurrency);
            double toCurrency = exchangeRates.get(convertCurrency);

            // perform conversion calculation
            double convertedValue = (amount / fromCurrency) * toCurrency;
            convertedValue = Math.round(convertedValue * 100.0) / 100.0; // round to 2 decimal places
            
            // format the result
            DecimalFormat formatter = new DecimalFormat("#,###.##");
            String formattedValue = formatter.format(convertedValue);

            // display the result
            resultLabel.setText("Converted Amount: " + formattedValue);
        } catch (NumberFormatException e) {
            resultLabel.setText("Invalid amount. Please enter a number.");// handle invalid input for amount
        }
    }
}
