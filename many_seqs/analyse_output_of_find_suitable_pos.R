library(tidyverse)

find_pairs <- function(data) {
  pairs <- c() # Initialize an empty list to store pairs
  
  for (i in seq_along(data)) {
    for (j in (i + 1):length(data)) {
      diff <- abs(data[i] - data[j])  # Calculate the absolute difference
      if (30 <= diff && diff <= 300 && diff %% 3 != 0 && !is.na(diff)) {
        pairs <- c(pairs, paste0(data[i], '-', data[j]))
      }
    }
  }
  
  return(pairs)
}


### Variables ###

commands_to_store <- 20  # this is the number of different attempts of SpliceNouveau that we will do
attempts_per_run <- 1
iterations_per_run <- 1000
csv_names <- c("data from find_suitable_position dot py/uniprotkb_proteome_UP000000625_AND_revi_2024_01_18.csv",
               "data from find_suitable_position dot py/uniprotkb_proteome_UP000005640_AND_revi_2024_01_18.csv")
CE_aim <- 0.5
intron1 = 'GTAAGAATGCACATCACTTCTTGAGAGTATGGAGGAGTGAAATGACACTCATGCCAGAGTTACTGTATGTCTGCACTTTAAAAGTGTAGCTTTTAAAAGATGATGAATGACTGTCTGTTTGTGTGTGTGTGTGTGaaTGTGTGTGTGTGTGTGTGTcacccAG'
intron2 = 'GTatgcatgactgcctgcatgcttgtttgttTGTTTGTATGTTTGTATGGAGTCGGGGTTTCTGAATGTATGCCTGAGGCTGGTTGCAGAGTCTCGCTCTGGATGTCTACGCTGGATGTGCAGTAACATGAGCCACTGTGCCCGGCCAATCCTAAGAATTTCTTTTGCGGTGGTTGCAAGTCTGGGCAGAACTCTTGTCAGGGGCTGTAACTGGACTTATCTTTACTCCTTTGTCAG'
five_utr <- 'tcgccacc'
three_utr <- 'TAAACAAATGGTAAGGAAGGGCACATCAATCTTTGCTTAATTGTCCTTTACTCTAAAGATGTATTTTATCATACTGAATGCTAAACTTGATATCTCCTTTTAGGTCATTGATGTCCTTCACCCCGGGAAGGCGACAGTGCCTAAGACAGAAATTCGGGAAAAACTAGCCAAAATGTACAAGACCACACCGGATGTCATCTTTGTATTTGGATTCAGAACTCAGTAAACTGGATCCGCAGGCCTCTGCTAGCT'

total_prots_to_try <- 100

##### RUN ######

for(csv_name in csv_names){

  proteome_name <- word(word(csv_name, 2, sep='/'), 1, sep='\\.')
  
  setwd("/Users/ogw/Library/CloudStorage/GoogleDrive-oscargwilkins@gmail.com/My Drive/UCL PhD/Year 5/Paper revisions/run on tons of sequences/")
  
  df <- read_csv(csv_name) %>%
    mutate(don_score = as.numeric(don_score),
           acc_score = as.numeric(acc_score),
           intron_position = as.numeric(intron_position)) %>%
    filter(!is.na(don_score)) %>%
    filter(!is.na(acc_score)) %>%
    filter(!is.na(intron_position)) %>%
    mutate(name = str_replace_all(name, "[^A-Za-z0-9]", "_"))
  
  prots_to_try_df <- df %>%
    distinct(name) %>%
    slice_head(n = total_prots_to_try)
  
  df <- df %>% 
    filter(name %in% prots_to_try_df$name)
  
  protein_seq_df <- df %>%
    filter(!is.na(sequence))
  
  for(this_name in prots_to_try_df$name){
    df2 <- df %>%
      filter(name == this_name) %>%
      filter(don_score > quantile(don_score, 0.5)) %>%
      filter(acc_score > quantile(acc_score, 0.5))
    
    pairs <- find_pairs(df2$intron_position)
    
    df3 <- data.frame(p1 = as.numeric(word(pairs, 1, sep="-")),
                      p2 = as.numeric(word(pairs, 2, sep="-"))) %>%
      sample_n(min(commands_to_store, length(pairs)))
    
    df4 <- df3 %>%
      mutate(command = paste0('sbatch -t 12:00:00 --cpus-per-task=16 --mem=16G -p gpu --gres=gpu:1 --wrap="',
                              'python3 SpliceNouveau_for_ht.py --initial_cds ',
                              protein_seq_df$sequence[which(protein_seq_df$name == this_name)],
                              ' --initial_intron1 ',
                              intron1,
                              ' --initial_intron2 ',
                              intron2,
                              ' --ce_start ',
                              p1,
                              ' --ce_end ',
                              p2, 
                              ' --five_utr ',
                              five_utr,
                              ' --three_utr ',
                              three_utr,
                              ' -a ',
                              attempts_per_run,
                              ' -n ',
                              iterations_per_run,
                              ' --ignore_end ',
                              str_length(intron2),
                              ' --target_cryptic_donor ',
                              CE_aim,
                              ' --target_cryptic_acc ',
                              CE_aim,
                              ' --n_seqs_per_it 1 --initialisations 3 --output ',
                              'auto_results/',
                              this_name, "Position_", p1, "_", p2, '_',
 
                              '"')) %>% 
      select(command) %>%
      mutate(protein_name = this_name)
    
    if(this_name == prots_to_try_df$name[1]){
      command_df <- df4 
    } else {
      command_df <- bind_rows(command_df, df4)
    }
  }
  
  write_csv(command_df, paste0(proteome_name, '_commands.csv.gz'))
}

