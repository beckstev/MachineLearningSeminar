<!-- * Plots der Ergebnisse anpassen bzw. updaten -->
<!-- * Loss- und Acc von MiniDogNN mit 120 Klassen -->
<!-- * Tabelle mit RF Hyperparametern in den Anhang, Genauigkeit und so auch? -->
<!-- * Loss und Acc Plot von MiniDogNN anpassen -->
* Schnauzenform und Fellfarbe bei visualize_predicitons einbringen?
* Ausblick mit dictionaries
<!-- * Plots prüfen -->

# Stevens Anmerkungen
## Einleitung
<!-- * nicht nur CNN genommen -->
<!-- * nicht unterschieden, sondern klassifiziert -->
<!-- * Satz kürzen -->
<!-- * Farbinformationen Klassifikation verbessern -->
<!-- * Weiterhin wird erörtert, wie ein vortrainiertes Netz beitragen kann oder so -->
<!-- * vielleicht kommt weg, sogar auch -->
<!-- * "gleiche" ändern -->
<!-- * unter anderem weg -->
<!-- * letzten teilsatz wegstreichen -->

## Datensatz
<!-- * Mit Dem Datenstaz -->
<!-- * minimal und maximal weg -->

## Lösungsansatz
<!-- * Strategie? -->
<!-- * Absatz? -->
<!-- * aufgrund der Verwendung von numpy arrays -->
<!-- * statt resizen reskalieren -->
<!-- * batchweise statt batchwise -->
<!-- * Scatter plots in den Anhang -->
<!-- * technische Schwierigkeiten raus -> einfach schreiben, dass auf 224x224 reresized wurde -->
* eventuell auf Doku verweisen
<!-- * fällt dies nicht ins Gewicht -->
<!-- * nebenbei raus und der Bilder auch -->
<!-- * statt CNN Architekturen -->
<!-- * da die Bilder... raus -->
<!-- * Freie Parameter Anzahl ändern -->
<!-- * um die Anzahl der freien Parameter für das FCN LAyer festzulegen -->
<!-- * Klammern bei PreBigDog -->
<!-- * schreiben, dass PreDog für 5 Klassen und PreBigDog für 120 Klassen -->
<!-- * Dazudichten: Auswirkungen auf Overtraining zu untersuchen -->
<!-- * PreLU Satz kaputt -->
<!-- * softmax ist normiert -->
<!-- * eigener Absatz -->
<!-- * nicht Anzahl Parameter reduziert, Bilduaflösung reduziert -->
<!-- * mit reshape wieder als Tensor -->
<!-- * Bilder sind normiert zwischen 0 und 1, und deswegen sigmoid, weil Output zwischen 0 und 1 -->
<!-- * Hyperparametern weglassen -->
<!-- * Cross Validation streichen -->
<!-- * Es wurden dreiSkalierungmeter optimiert, learning rate und epochen nicht, da early stopping -->
<!-- * default werte nennen -->
<!-- * Um den Informationsgehalt zu prüfen, statt wie wichtig -->
<!-- * einzelne Absätze -->
<!-- * für den gegebenen Zeitraum -->
* Alternativ-Methode Sklaierung 96x96 verwenden?

## Ergebnisse
<!-- * Dann erst MiniDogNN -->
<!-- * Hyperparamter als Einleitung mach ich nicht -->
<!-- * Folgerichtig optimiert Farbinformationen die Genauigkeit -->
<!-- * Statt Neuronale Netze Architekturen, 60% nur auf PreDog -->
<!-- * dann weg -->
<!-- * für den großen Datensatz ungefähr 1000 Bilder zu klein, deswegen verwenden -->
<!-- * möglich klein -->
<!-- * Hyperparameter weg lassen -->
<!-- * Confusion Matrix Schreibweise -->
<!-- * was für eine hohe Genauigkeit spricht -->
<!-- * noch weg -->
<!-- * beider Klassen weg -->
<!-- * alles in allem -> Zusammenfassend zeigt eine hohe Genauigkeit -->
<!-- * Prozente African Hunting Dog -->
<!-- * Nur 64 Feature aus CNN, und damit dann 120 Klassen klassifizieren -->
<!-- * vspace nutzen -->
<!-- * noch raus -->
<!-- * statt besser verläuft präziser oder so -->
<!-- * Random Forest ausschreiben -->
<!-- * statt 120 Klassen: Auf dem gesamten Datensatz -->
<!-- * Skalierungsproblem der Achsen -->
<!-- * Architekturen dazu schreiben -->

## Zusammenfassung
<!-- * Bildklassifikation -->
<!-- * deutliche Verbesserung der Genauigkeit -->
<!-- * Alternative Methode: Outperformed MiniDogNN, ist aber wegen zu wenig Parametern -->
<!-- den vortrainierten Netzen unterlegen -->
<!-- * Alles in allem -->
<!-- * in Sachen -->
<!-- * vermuten, dass besser -->
* wurden durchsuchen
