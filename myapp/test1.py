import tf_idf_summary
import glob

data='''A plethora of bibliometric indicators is available nowadays to gauge research performance. The spectrum of bibliometric based measures is 
very broad, from purely size-dependent indicators (e.g. raw counts of scientific contributions and/or citations) up to size-independent
measures (e.g. citations per paper, publications or citations per researcher), through a number of indicators that effectively combine 
quantitative and qualitative features (e.g. the h-index). In this paper we present a straightforward procedure to evaluate the scientific
contribution of territories and institutions that combines size-dependent and scale-free measures. We have analysed in the paper the 
scientific production of 189 countries in the period 2006–2015. Our approach enables effective global and field-related comparative
analyses of the scientific productions of countries and academic/research institutions. Furthermore, the procedure helps to identifying 
strengths and weaknesses of a given country or institution, by tracking variations of performance ratios across research fields. Moreover,
by using a straightforward wealth-index, we show how research performance measures are highly associated with the wealth of countries and 
territories. Given the simplicity of the methods introduced in this paper and the fact that their results are easily understandable
by non-specialists, we believe they could become a useful tool for the assessment of the research output of countries and institutions.
'''
summary=tf_idf_summary.summary1(data)
print(1)
print(summary[0])
