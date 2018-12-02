#!/usr/bin/env python3

# '+' indicates index at which to place tabular data
table_template = r'\begin{table}[H] \
\tablecaption{TODO caption}        \
\label{TODO label}                 \
\resizebox{\textwidth}{!}{%        \
\begin{tabular}{|l|l|l|}           \
\hline               \
  & + & +  \\ \hline \
 + & + & + \\ \hline \
 + & + & + \\ \hline \
 + & + & + \\ \hline \
 + & + & + \\ \hline \
 + & + & + \\ \hline \
 + & + &   \\ \hline \
 + & + &   \\ \hline \
 + &  &    \\ \hline \
  &  &     \\ \hline \
  + & + &  \\ \hline \
  + & + &  \\ \hline \
  + & + &  \\ \hline \
  + & + &  \\ \hline \
  + & + &  \\ \hline \
  + &  &   \\ \hline \
  \end{tabular}%     \
}                    \
\end{table} '

# Remove backslashes from table_template
table_template = '\n'.join([line[:-1] for line in table_template.split('\n')])
                    
with open('stats.txt', 'r') as f:
    results = f.read().split('\n\n\n')[:-1]

    with open('stats_tables.txt', 'w') as output_f:
        for raw_result in results:
            splits = raw_result.split('\n')

#            for split in splits:
#                split += ','

            result = []

            for split in splits:
                result += [item.strip() for item in split.split(',')]

            result = [item for item in result if item]

            for res_index in range(len(result)):
                split_item = result[res_index].split()
                for word_index in range(len(split_item)):
                    if split_item[word_index][-1] == '_':
                       split_item[word_index] = split_item[word_index][:-1]

                result[res_index] = ' '.join(split_item)

            output_table = table_template

            result_index = 0

            next_insertion_index = output_table.find('+')
            while next_insertion_index >= 0:
                output_table = output_table[:next_insertion_index] + result[result_index] + \
                    output_table[next_insertion_index + 1:]
                
                result_index += 1
                next_insertion_index = output_table.find('+')

            if 'nor' in output_table:
                find_str1 = 'nor'
                find_str2 = 'is statistically better'

                nor_index = output_table.find(find_str1)
                output_table = output_table[:nor_index + len(find_str1)] + ' & & \\\\\n' + \
                    output_table[nor_index + len(find_str1):]

            output_f.write(output_table.replace('_', r'\_'))
            output_f.write('\n\n')
