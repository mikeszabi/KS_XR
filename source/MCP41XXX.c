///////////////////////////////////////////////////////////////////////////
////   Library for a mcp41xxx		                                   ////
////                                                                   ////
////   set_pot (int data);  Sets pot to new_value                      ////
////                                                                   ////
////   shutdown_pot ();  shutdown pot to save power                    ////
////                                                                   ////
///////////////////////////////////////////////////////////////////////////
#define POT_A    1 
#define POT_B    2 
#define POT_BOTH 3 

void set_pot(int data,int POT) {

	BYTE cmd=0x11;
	
   if (POT==POT_A) cmd = 0b00010001; 
   if (POT==POT_B) cmd = 0b00010010; 
   if (POT==POT_BOTH) cmd = 0b00010011; 

    output_low(FILTCS);
  	spi_write2(cmd);
  	spi_write2(data);
    output_high(FILTCS);
}

void shutdown_pot() {
  	BYTE cmd=0x21;
    output_low(FILTCS);
  	spi_write2(cmd);
    output_high(FILTCS);
}
