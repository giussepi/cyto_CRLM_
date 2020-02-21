## Folder Structure:
	For more details read the README file inside each folder.

1. Data 
	-->  Preprocessing data - patching croping
	-->  Create DataLoader for pytorch

2. Utils
	-->  All kinds of shared tools such as visualization
	-->  Anonymize dataset

3. Models
	--> Define the models for the 

4. Options
	-->  Parameters for training
	

5. Train
	-->   Train modules and scripts


6. Load
	-->  Load the trained models
	-->  Use Sliding window to get the whole slide results
	-->  All kinds of post processing trails


## Note on Data Location
	
### WSIs  DATASET from KI :
		CRLM :  
			 all HE slides   -->    robbin:/mnt/DATA_CRLM/CRLM/Original/
			 HE / IHC  slides  -->  robbin:/mnt/DATA_CRLM/CRLM/IHC_HE/ALL/

		DCIS : Camelyon:/mnt/DATA_DCIS/DCIS_original/


		CRC-LND :  Camelyon:/mnt/DATA_CRLM/CRC_LND/


		Pancreatic cancer :  /mnt/DATA_OTHER/Pancreas/  
							--> Check if it is anynomized?????

### Patches for John's Code:
		Tissue Level (Ch3) :   robbin:/mnt/DATA_CRLM/Patches/

		Saved model location:   robbin:/mnt/DATA_CRLM/Checkpoint




## Note on how to use server

------>>>>> Remeber: Do not put Data in your home folder.
					 Try to link command (ls -n ) link your folder from Data Disk  to your own home
