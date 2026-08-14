# Autoencoder: Metriken zur Bewertung des Trainings

Zur Bewertung des Trainings eines Autoencoders eignen sich verschiedene Metriken, je nachdem, welchen Aspekt du beurteilen möchtest.

## 1. Rekonstruktionsfehler (Kernmetrik)

Das Wichtigste ist meist der Rekonstruktionsverlust – also wie gut der Decoder die Eingabe aus der latenten Repräsentation wiederherstellt.

| Metrik | Einsatzbereich |
|--------|----------------|
| **MSE (Mean Squared Error)** | Standard bei kontinuierlichen Daten |
| **MAE (Mean Absolute Error)** | Robuster gegen Ausreißer |
| **Binary Cross-Entropy (BCE)** | Bei binären oder auf [0,1] normalisierten Daten (z. B. MNIST) |
| **Huber Loss** | Kompromiss zwischen MSE und MAE |

## 2. Bildspezifische Qualitätsmaße

Wenn du Bilder rekonstruierst, sagen MSE-Werte oft wenig über die wahrgenommene Qualität aus:

- **PSNR (Peak Signal-to-Noise Ratio)** – klassisches Maß für Rekonstruktionsqualität
- **SSIM (Structural Similarity Index)** – berücksichtigt Struktur statt nur Pixelabweichung
- **LPIPS** – nutzt vortrainierte Netze für perzeptuelle Ähnlichkeit
- **FID (Fréchet Inception Distance)** – vor allem bei generativen Varianten wie VAEs interessant

## 3. Spezifisch für Variational Autoencoder (VAE)

Hier setzt sich der Loss aus zwei Komponenten zusammen, die man einzeln beobachten sollte:

- **ELBO (Evidence Lower Bound)** als Gesamtmetrik
- **KL-Divergenz** zwischen latenter Verteilung und Prior
- **Rekonstruktionsterm** separat, um *posterior collapse* zu erkennen (wenn die KL gegen 0 geht und der Decoder den Latent ignoriert)

## 4. Qualität des latenten Raums

Ein guter Rekonstruktionsfehler heißt nicht automatisch, dass die Repräsentation nützlich ist:

- Visualisierung mit **t-SNE** oder **UMAP**, um Cluster/Struktur zu prüfen
- **Disentanglement-Metriken** wie MIG, DCI oder der β-VAE-Score
- **Klassifikations- oder Clustering-Performance** auf den Latent Features (Linear Probing) als Downstream-Test

## 5. Generalisierung und Anwendungsfall

- Lücke zwischen **Trainings- und Validierungsloss** (Overfitting-Diagnose)
- Bei **Denoising-Autoencodern**: Rekonstruktionsqualität aus verrauschten Eingaben
- Bei **Anomalie-Detektion**: AUC-ROC/PR auf dem Rekonstruktionsfehler als Anomalie-Score
- Bei **Kompression**: Rate-Distortion-Kurven (Bits pro Dimension vs. Fehler)

## Praktischer Tipp

Verlass dich nicht nur auf eine einzige Zahl. Kombiniere den quantitativen Loss immer mit **visueller Inspektion** einiger Rekonstruktionen – gerade bei Bildern siehst du sofort Artefakte wie Weichzeichnung oder Modus-Kollaps, die MSE alleine verschleiert.

---

## Übersicht: Welche Metrik wann?

| Autoencoder-Typ | Primäre Metriken | Zusätzlich sinnvoll |
|-----------------|------------------|---------------------|
| Klassisch (Bilder) | MSE, SSIM, PSNR | LPIPS, visuelle Inspektion |
| Klassisch (Tabelle) | MSE, MAE | Feature-weise Fehleranalyse |
| VAE | ELBO, KL, Rekonstruktionsterm | FID, Disentanglement-Metriken |
| Denoising | MSE/SSIM auf sauberem Ziel | PSNR-Gewinn ggü. Input |
| Anomalie-Detektion | AUC-ROC, AUC-PR | F1, Precision@k |
| Kompression | Rate-Distortion-Kurve | Bits per Pixel/Dimension |
