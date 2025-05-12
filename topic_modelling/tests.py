from django.test import TestCase
import re

corpus = """Yorum 1 (2015-01-22T16:33:14+0300): BİSMİLLAHİRRAHMANİRRAHİMarkadaşlarım daha önceki topiklerde yeri geldi sevindik yeri geldi üzüldük.Bunlara rağmen hiç umudumuzu kaybetmedik.Hadi bu başlık altında şubat ayında bu heyecanlı süreci geçirecek olan anne adaylarını sohbete bekliyorum.Güncelleme: Arkadaşlar aralık geçti hadi neyse dedim,ocak geçti hadi onuda kabul..şubata erteleyelim dedik.şimdi şubatta da olmayacak.Tüp işi başka mani çıkmazsa mart ayında başlayacak.Artık hiçbişey için kesin konuşmuyorum..inşallah sizler bu ay anne olursunuz.&Yorum 2 (2015-01-22T16:33:14+0300): Hayırlı uğurlu olsunnnnher gelen hamiş olsunnn insallahhhhben transferim için regl olmayı bekliyorum aysonuna dogru olacak transferde subat 10 falan olur heralde&Yorum 3 (2015-02-04T12:11:38+0300): hadi hayırlısı canım.bende bi aksilik olmazsa (ki inşallah) aralık ocak olmadı ertelemek zorunda kaldık.inşallah şubatta olur benimkide.senin içinde hayırlısı olsun inşallah&Yorum 4 (2015-01-22T16:41:44+0300): ben de varımmm hayırlı uğurlu olsun topiğin inşallah canım cumartesi histeroskopim var umarım şubat başı adetimle başlarız tedaviye. hepimizin yüzü gülsün artık aminnn:)&Yorum 5 (2015-01-22T16:57:28+0300): amin canım hoş geldin topiğimize..yeni başlıklar yeni umutlar hepimizin olsun inşallah&Yorum 6 (2015-01-22T17:35:09+0300): Bana sonrasinda histeroskopi tecrubeni anlatirmisi n simdiden gecmis olsun guzel haberler pesinden gelsin&"""

# Match all content after each "Yorum X (date): " up to the next &
pattern = r"Yorum \d+ \([^)]+\): (.*?)(?=&|$)"
matches = re.findall(pattern, corpus, re.DOTALL)

# Clean up any leading/trailing spaces
corpus = [match.strip() for match in matches]
for sentence in corpus:
    print(f'{sentence}\n')



