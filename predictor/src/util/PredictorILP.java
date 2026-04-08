/*
    PredictorILP.java
    Clase que inicia procesos de prediccion de ORFs para juego de secuencias dadas. 

    Copyright (C) 2016 
    Jose Lopez (jlopez@unet.edu.ve), Samuel Useche (maverick71036@gmail.com).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License asSamuel Useche (maverick71036@gmail.com)
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>

*/

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package util;

/**
 *
 * @author Jose Lopez (jlopez@unet.edu.ve), Samuel Useche (maverick71036@gmail.com).  
 */
public class PredictorILP {

    public static void main(String[] args) throws Exception {
        
        //String genesEnProceso = "salidas/pruebasPaper.txt";
        String genesEnProceso = "entrada.txt";
        String idsGenesEnProceso = "ensemblIDs.txt"; 
        String salidaPredGFF3 = "salidas/salidaPredictor.gff3";
        String salidaEnsEPDGFF3 = "salidas/salidaEnsblEPD.gff3";

        // ---este constructor que esta comentado solo recibe la hebra ----
        //GenInformation obj = new GenInformation("+"); 
        //---- este constructor recibe cuales lecturas van a tomarse para generar el reporte GTF. Resulta que una secuencia
        // puede tener mas de una lectura (estructura) asociada. Se indica ademas si el reporte GTF va a incluir
        // los atg, las paradas, los exones y los intrones----. El argumento "completo" indica que se genere reporte GTF
        // para cada lectura obtenida para la secuencia problema.
        GenInformation generador = new GenInformation(args[15], "1", true, true, false, true, Boolean.parseBoolean(args[11]), Boolean.parseBoolean(args[12]), Boolean.parseBoolean(args[13]), Boolean.parseBoolean(args[14]));
     

        //------llamamos al inicio al cual le vamos a enviar los url de los archivos ---
        generador.inicio(genesEnProceso, idsGenesEnProceso, salidaPredGFF3, salidaEnsEPDGFF3, args[0], args[1], args[2], args[3], Boolean.parseBoolean(args[4]), Boolean.parseBoolean(args[5]), Boolean.parseBoolean(args[6]), Integer.parseInt(args[7]), Integer.parseInt(args[8]), Boolean.parseBoolean(args[9]), args[10]);

    }
}
