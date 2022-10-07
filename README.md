## Leak current, even with gigaohm seals, can cause misinterpretation of stem-cell derived cardiomyocyte action potential recordings 

This repository contains all code, data, and figures for our manuscript: "Leak current, even with gigaohm seals, can cause misinterpretation of stem-cell derived cardiomyocyte action potential recordings"

By: Alexander P. Clark, Michael Clerx, Siyu Wei, Chon Lok Lei, Teun P. de Boer, Gary R. Mirams, David J. Christini, Trine Krogh-Madsen

The code requires Python 3.7+ and two dependencies: Myokit and DEAP. These can be installed using `pip install myokit` and `pip install deap`. Alternatively, you can `pip install -r requirements.txt` to install all dependencies at once. 

#### Files and folders

- `fX-*.py` – Each of the numbered files contain the code to reproduce their corresponding figure. For example, `f9-fit-base-to-leak.py` will first run a GA that fits the Kernik model to Kernik+leak, and then plots Figure 9.
- `data/` – Folder with all experimental data and GA results.
- `mmt/` – Contains all Myokit model files  
- `figure-pdfs` – PDF files for all manuscript figures. 

As mentioned, `f9-fit-base-to-leak.py` contains the GA code used to produce the results in Figure 9. To run the GA on your own computer you must uncomment `fit_model()` in `main()`. The GA results are saved to `./data/ga_results/inds_bCa_bNa_fixed.pkl`. If the code fails or gets hung up when you run, try removing the multithreading. You can do this by commenting/uncommenting code on lines 70 through 74 which should look like:

```py
# To speed things up with multi-threading
#p = Pool()
#toolbox.register("map", p.map)

# Use this if you don't want multi-threading or get an error
toolbox.register("map", map)
```

The code will run slower, but this may resolve your issue.

#### Supporting materials 


