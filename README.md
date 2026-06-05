---

# README

Nella repository sono presenti i seguenti script Python:

* tre modelli utilizzati per decodificare le rappresentazioni ULIP2 in point cloud:
  * `decoder_v1.py`: MLP semplice.

  * `decoder_v2.py`: MLP con una struttura leggermente più complessa.

  * `decoder_v3.py`: modello basato su transformers.

* `train.py`: codice per il training del decoder. Utilizza la Chamfer Loss e una procedura di subsampling per ridurre l’utilizzo di memoria GPU.

* `dataloader.py`: script per il caricamento del dataset.

* `pc_transform.py`: a partire da una point cloud e da una coppia di testi che definiscono un polo positivo e uno negativo di una determinata caratteristica:

  1. calcola l’embedding della point cloud tramite un modello ULIP2;
  2. calcola la direzione nello spazio latente definita dai due testi in 5 modalità diverse:
     I: direzione calcolata tra i due poli testuali (o delle pointcloud/mesh di esempio)
     II: direzione calcolata tra i baricentri degli embedding più spostati verso i poli
    III: direzione calcolata tra polo positivo e embedding di partenza
     IV: direzione calcolata tra baricentro dei punti più vicini al polo positivo e embedding
     V: direzione calcolata tra il baricentro dei punti più vicini al polo positivo e quello dei punti più lontani dal polo positivo
  3. se gli embedding dei poli sono già presenti, evita di ricaricare il checkpoint ULIP2 (operazione che richiede alcuni minuti);
  4. sposta linearmente l’embedding lungo tale direzione tramite il parametro `ALPHA`;
  5. normalizza nuovamente il punto sull’ipersfera latente;
  6. se è stato scelto un valore `ADAPT_STEPS > 0`, viene inoltre eseguito un fine-tuning del checkpoint del decoder sulla specifica istanza da modificare;
  7. viene generata la point cloud corrispondente al nuovo punto nello spazio latente applicando uno spostamento sull'ipersfera con SLERP;
  8. il risultato viene salvato nella stessa cartella della point cloud originale, con un nome che include:
      * valore di `ALPHA`;
      * polo positivo e negativo (se presente);
      * modalità di calcolo della direzione;
      * numero di `ADAPT_STEPS`.

Nelle cartelle:

* `transform_input_face`
* `transform_input_donkey`
* `transform_input_spot`

sono presenti alcuni esempi di esecuzione dello script `pc_transform.py`, ottenuti a partire da diverse point cloud iniziali e con differenti valori di `ADAPT_STEPS` e `ALPHA`.
