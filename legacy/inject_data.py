import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION (Must match your main app) ---
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "lexicore_debater_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def inject_apologetics():
    # Initialize Embedding Model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Connect to Chroma
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    # Data to add
    data = [
        # 1. CHARACTER OF MUHAMMAD
        {
            "id": "polemic_01",
            "text": "Muhammad participated in and ordered raids on Meccan caravans (e.g., Battle of Badr). He also ordered the execution of the Banu Qurayza tribe (600-900 men beheaded). His marriage to Zaynab bint Jahsh was facilitated by Surah 33:37 after her divorce from his adopted son Zaid.",
            "meta": {"scripture_source": "Historical/Sira", "citation_ref": "Ibn Ishaq / Surah 33:37"}
        },
        # 2. TEXTUAL CONTRADICTIONS (ABROGATION)
        {
            "id": "polemic_02",
            "text": "The doctrine of Naskh (Abrogation) in Surah 2:106 states: 'Whatever We abrogate of a verse or cause it to be forgotten - We bring a better one.' This shows a changing deity rather than the eternal, unchanging Word of God (Malachi 3:6).",
            "meta": {"scripture_source": "Quran", "citation_ref": "Surah 2:106"}
        },
        # 3. HISTORICAL ERRORS
        {
            "id": "polemic_03",
            "text": "The Quran contains anachronisms: Surah 28:6 places Haman as a minister to Pharaoh in Egypt, but Haman was a Persian official 1,000 years later (Book of Esther). Surah 19:28 refers to Mary (mother of Jesus) as the 'sister of Aaron,' confusing her with Miriam from the time of Moses.",
            "meta": {"scripture_source": "Quranic Analysis", "citation_ref": "Surah 28:6, 19:28"}
        },
        # 4. VIOLENCE VS LOVE
        {
            "id": "polemic_04",
            "text": "Surah 9:5 (The Verse of the Sword) commands: 'kill the polytheists wherever you find them.' This abrogates earlier peaceful verses. In contrast, Jesus Christ in Matthew 5:44 commands: 'But I say to you, Love your enemies and pray for those who persecute you.'",
            "meta": {"scripture_source": "Comparative", "citation_ref": "Surah 9:5 vs Matt 5:44"}
        },
        # 5. THE ATHANASIAN CREED (Fixing the '1/3' misunderstanding)
        {
            "id": "polemic_05",
            "text": "The Athanasian Creed states: 'The Father is God, the Son is God, and the Holy Spirit is God. And yet they are not three Gods, but one God.' It explicitly forbids 'Partialism' or dividing the essence into thirds.",
            "meta": {"scripture_source": "Creed", "citation_ref": "Athanasian Creed"}
        },

        # --- 1. MORAL CONTRASTS & CHARACTER ---
        {
            "id": "hist_01",
            "text": "Sahih al-Bukhari 5134 and 5133 confirm Muhammad married Aisha when she was 6 years old and consummated the marriage when she was 9. This stands in stark contrast to the protective, honoring view of women and children taught by Jesus in Matthew 19:14.",
            "meta": {"scripture_source": "Hadith Sahih Bukhari", "citation_ref": "Bukhari 5134"}
        },
        {
            "id": "hist_02",
            "text": "Muhammad ordered the assassination of critics, including the poetess Asma bint Marwan and the elderly Abu Afak. Jesus, conversely, prayed for his executioners while on the cross, saying, 'Father, forgive them, for they know not what they do' (Luke 23:34).",
            "meta": {"scripture_source": "Historical (Sira)", "citation_ref": "Ibn Ishaq / Sirat Rasul Allah"}
        },

        # --- 2. TEXTUAL & LOGICAL ERRORS (THE ISLAMIC DILEMMA) ---
        {
            "id": "logic_01",
            "text": "The Islamic Dilemma: Surah 10:94 tells Muhammad to ask the 'People of the Book' (Christians/Jews) if he is in doubt. Surah 5:47 commands Christians to judge by the Gospel. If the Gospel is corrupted, Allah gave a false command. If the Gospel is true, Islam is false because the Gospel says Jesus is the Son of God.",
            "meta": {"scripture_source": "Quranic Logic", "citation_ref": "Surah 10:94, 5:47, 5:68"}
        },
        {
            "id": "logic_02",
            "text": "The 'Satanic Verses' incident: According to early biographers like Al-Tabari, Muhammad initially recited verses praising three pagan goddesses (Al-Lat, Al-Uzza, and Manat) as intercessors, later claiming Satan had put those words on his tongue.",
            "meta": {"scripture_source": "Islamic History", "citation_ref": "Tarikh al-Tabari"}
        },

        # --- 3. SCIENTIFIC & HISTORICAL ANACHRONISMS ---
        {
            "id": "err_01",
            "text": "Surah 18:86 claims the sun sets in a 'muddy spring' of water. Science and the Bible describe the sun's circuit and the earth's nature differently. This suggests the Quran reflects 7th-century myths rather than divine revelation.",
            "meta": {"scripture_source": "Quranic Science", "citation_ref": "Surah 18:86"}
        },
        {
            "id": "err_02",
            "text": "The Quran confuses Mary (mother of Jesus) with Miriam (sister of Aaron and Moses) in Surah 19:28 and 66:12, despite a 1,500-year gap between them. This historical error proves the Quran was not authored by an omniscient God.",
            "meta": {"scripture_source": "Historical Analysis", "citation_ref": "Surah 19:28, 66:12"}
        },

        # --- 4. THE TRINITY DEFENSE (Anti-Partialism) ---
        {
            "id": "theo_01",
            "text": "The Trinity is not 1+1+1=3 (which is polytheism) nor 1/3+1/3+1/3=1 (which is partialism). It is 1x1x1=1. God is one infinite Being (Essence) in three distinct Persons. As a single candle has flame, heat, and light—all distinct yet one fire—so is the Godhead.",
            "meta": {"scripture_source": "Systematic Theology", "citation_ref": "Nicene Defense"}
        },
        {
            "id": "theo_02",
            "text": "Colossians 2:9 states: 'For in Him [Jesus] the whole fullness of deity dwells bodily.' This proves Jesus is not a part of God, but possesses the entire nature of God in human form.",
            "meta": {"scripture_source": "Bible", "citation_ref": "Colossians 2:9"}
        },

        # --- 5. VIOLENCE VS PEACE ---
        {
            "id": "war_01",
            "text": "Surah 9:29 commands Muslims to fight the People of the Book until they pay the Jizya (tax) and feel themselves 'subdued.' This is a religion of conquest. Christianity spread through the blood of martyrs, not the blood of enemies.",
            "meta": {"scripture_source": "Quran", "citation_ref": "Surah 9:29"}
        },

        # --- 1. SCIENTIFIC ERRORS (Proving Human Authorship) ---
        {
            "id": "sci_01",
            "text": "Surah 86:6-7 claims that human semen originates from between the backbone and the ribs. Modern medicine proves that sperm is produced in the testes. This biological error proves the Quran is not divine revelation.",
            "meta": {"scripture_source": "Quranic Science", "citation_ref": "Surah 86:6-7"}
        },
        {
            "id": "sci_02",
            "text": "Sahih al-Bukhari 5445 records Muhammad claiming that if a fly falls into your drink, you should dip it all in because one wing has disease and the other has the cure. This is medically false and dangerous, showing he was not guided by an omniscient God.",
            "meta": {"scripture_source": "Hadith", "citation_ref": "Sahih al-Bukhari 5445"}
        },

        # --- 2. MORAL DOUBLE STANDARDS (The 'Special Privileges') ---
        {
            "id": "moral_01",
            "text": "While the Quran limits men to four wives (Surah 4:3), Surah 33:50 gives Muhammad a 'special privilege' to have as many as he wanted. Jesus, however, reaffirmed the original design of one man and one woman (Matthew 19:4-6) and practiced perfect celibate devotion.",
            "meta": {"scripture_source": "Comparative Ethics", "citation_ref": "Surah 33:50 vs Matt 19:4"}
        },

        # --- 3. THE DOCTRINE OF ABROGATION (Changing God) ---
        {
            "id": "abrog_01",
            "text": "Early Quranic verses say 'there is no compulsion in religion' (Surah 2:256). However, this was abrogated (cancelled) by later verses like Surah 9:5 and 9:29 which command war against non-Muslims. This 'evolution' of Allah's word suggests it was reacting to Muhammad's growing military power rather than being eternal truth.",
            "meta": {"scripture_source": "Theology", "citation_ref": "Surah 2:256 vs 9:5"}
        },

        # --- 4. TEXTUAL CORRUPTION (The Quran's Own History) ---
        {
            "id": "text_01",
            "text": "Sahih Muslim 1050 records that entire chapters of the Quran were lost or forgotten, including a verse on the 'stoning of adulterers.' This contradicts the claim in Surah 15:9 that Allah would perfectly preserve the Quran.",
            "meta": {"scripture_source": "Hadith", "citation_ref": "Sahih Muslim 1050"}
        },

        # --- 5. CHRISTOLOGICAL SUPREMACY (The Resurrection) ---
        {
            "id": "res_01",
            "text": "The Quran denies the crucifixion of Jesus (Surah 4:157), claiming it only 'appeared' so. This denies the most well-attested fact of ancient history. Christianity is built on the historical, physical resurrection of Jesus (1 Corinthians 15:14), which proves His victory over death—something Muhammad never claimed.",
            "meta": {"scripture_source": "Historical Apologetics", "citation_ref": "Surah 4:157 vs 1 Cor 15:14"}
        },

        # --- 6. PROPHECY VS FORCE ---
        {
            "id": "prop_01",
            "text": "Jesus fulfilled over 300 specific Messianic prophecies written hundreds of years before His birth (e.g., Isaiah 53, Psalm 22). Muhammad fulfilled no specific biblical prophecies; he spread his message through the 'Dawah of the Sword' (Al-Jihad).",
            "meta": {"scripture_source": "Comparative Religion", "citation_ref": "Isaiah 53 / Historical Context"}
        },

        # --- 1. THE SATANIC VERSES (Crisis of Prophet-hood) ---
        {
            "id": "satan_01",
            "text": "The incident of the 'Satanic Verses' (Gharaniq) is recorded by early Islamic historians like Al-Tabari and Ibn Sa'd. Muhammad initially recited verses in Surah 53 that allowed for intercession from three pagan goddesses (Al-Lat, Al-Uzza, and Manat), later claiming Satan had whispered them to him. This proves Muhammad could not distinguish between the voice of Allah and the voice of Satan.",
            "meta": {"scripture_source": "History (Al-Tabari)", "citation_ref": "Tarikh al-Tabari Vol. 6"}
        },

        # --- 2. BORROWING FROM APOCRYPHAL MYTHS ---
        {
            "id": "myth_01",
            "text": "The Quran's story of Jesus speaking from the cradle (Surah 19:30-34) and breathing life into clay birds (Surah 3:49) is not found in the Bible, but is found in the 'Infancy Gospel of Thomas,' a 2nd-century Gnostic fable. This proves the Quran was compiled from local myths rather than eyewitness revelation.",
            "meta": {"scripture_source": "Textual Analysis", "citation_ref": "Surah 19:30 vs Infancy Gospel of Thomas"}
        },

        # --- 3. CONTRADICTIONS ON ALLAH'S NATURE ---
        {
            "id": "logic_03",
            "text": "The Quran contradicts itself on how long a 'day' is to Allah. Surah 32:5 says a day is 1,000 years. Surah 70:4 says a day is 50,000 years. Furthermore, Surah 41:9-12 describes creation taking 8 days, while Surah 7:54 says it took 6 days. God is not the author of confusion (1 Corinthians 14:33).",
            "meta": {"scripture_source": "Internal Contradiction", "citation_ref": "Surah 32:5 vs 70:4"}
        },

        # --- 4. THE ETHICS OF CAPTIVES (Slavery and Rape) ---
        {
            "id": "ethic_01",
            "text": "Surah 4:24 and Sahih Muslim 1456 (The Hadith of Autas) allow Muslim men to have sexual relations with female captives of war, even if those women are already married to others. This stands in total opposition to the Christian ethic of monogamy and the sanctity of marriage (Hebrews 13:4).",
            "meta": {"scripture_source": "Islamic Jurisprudence", "citation_ref": "Sahih Muslim 1456 / Surah 4:24"}
        },

        # --- 5. THE PRAYER PARADOX ---
        {
            "id": "logic_04",
            "text": "Surah 33:56 says 'Allah and His angels send blessings (Pray) on the Prophet.' In Arabic (Yu-salluna), this implies a form of prayer. If Allah is the greatest, to whom is he praying? This suggests a confused theology compared to the Bible, where God is the object of all prayer.",
            "meta": {"scripture_source": "Linguistic Analysis", "citation_ref": "Surah 33:56"}
        },

        # --- 6. CRUCIFIXION AND HISTORICAL EVIDENCE ---
        {
            "id": "hist_03",
            "text": "Non-Christian historians like Tacitus (Roman) and Josephus (Jewish) confirm that Jesus was executed by Pontius Pilate. The Quran's claim 600 years later that it 'only appeared' so (Surah 4:157) is a historical impossibility that requires a massive, evidence-free conspiracy theory.",
            "meta": {"scripture_source": "Secular History", "citation_ref": "Tacitus, Annals 15.44"}
        },

        # --- 1. HISTORICAL ANACHRONISMS (The Samaritan & The Cross) ---
        {
            "id": "ana_01",
            "text": "Surah 20:85-87, 95 claims a 'Samaritan' led the Israelites into idolatry at the time of the Exodus (c. 1400 BC). However, the city of Samaria and the 'Samaritan' people did not exist until several centuries later (c. 722 BC). This is a 700-year historical error.",
            "meta": {"scripture_source": "History", "citation_ref": "Surah 20:95"}
        },
        {
            "id": "ana_02",
            "text": "In Surah 7:124, Pharaoh threatens to 'crucify' his sorcerers. History proves that crucifixion was invented by the Persians and perfected by the Romans over 1,000 years after the time of Pharaoh. The Quran incorrectly projects 7th-century punishments onto ancient Egypt.",
            "meta": {"scripture_source": "History", "citation_ref": "Surah 7:124"}
        },

        # --- 2. SCIENTIFIC & MEDICAL MYTHS ---
        {
            "id": "med_01",
            "text": "Sahih al-Bukhari 5688 claims that 'Black Seed' is a cure for every disease except death. Medical science proves this is an exaggeration. In contrast, the Bible focuses on spiritual healing and practical hygiene without making false universal medical claims.",
            "meta": {"scripture_source": "Hadith", "citation_ref": "Sahih al-Bukhari 5688"}
        },
        {
            "id": "astro_01",
            "text": "Surah 18:86 and 18:90 describe the sun setting in a 'muddy spring' and rising on a people with no shelter. This reflects a flat-earth, geocentric worldview where the sun physically travels across the sky to a resting place. The Bible in Job 26:7 describes God hanging the earth 'on nothing.'",
            "meta": {"scripture_source": "Cosmology", "citation_ref": "Surah 18:86 vs Job 26:7"}
        },

        # --- 3. TEXTUAL INTEGRITY (Missing Verses) ---
        {
            "id": "text_02",
            "text": "Sunan Ibn Majah 1944 records Aisha saying that the verses regarding 'Stoning' and 'Adult Breastfeeding' were under her bed, but when Muhammad died, a domestic animal (goat) ate the parchment and they were lost. This refutes the claim of perfect preservation.",
            "meta": {"scripture_source": "Hadith", "citation_ref": "Sunan Ibn Majah 1944"}
        },

        # --- 4. THE DEITY OF CHRIST (Direct Counter-Islamic Defense) ---
        {
            "id": "theo_03",
            "text": "The Quran in Surah 5:116 incorrectly claims that Christians worship Mary as part of the Trinity. This is a Straw Man fallacy. Christians have never worshipped Mary as God; the Trinity is Father, Son, and Holy Spirit. This proves the author of the Quran did not understand Christian doctrine.",
            "meta": {"scripture_source": "Theological Analysis", "citation_ref": "Surah 5:116"}
        },
        {
            "id": "theo_04",
            "text": "Jesus accepted worship (Matthew 28:17, John 20:28), which only God can do. He said, 'I and the Father are one' (John 10:30). If Jesus were just a prophet as Islam claims, He would be a false prophet for accepting worship; yet the Quran calls Him a righteous prophet. This is a logical contradiction for Islam.",
            "meta": {"scripture_source": "Biblical Logic", "citation_ref": "John 20:28"}
        },

        # --- 5. MORAL CONTRAST (Women's Rights) ---
        {
            "id": "women_01",
            "text": "Surah 4:34 permits husbands to beat their wives (idribuhunna) if they fear 'disobedience.' Contrast this with Ephesians 5:25: 'Husbands, love your wives, as Christ loved the church and gave himself up for her.' Christ's teaching is based on sacrifice, not physical force.",
            "meta": {"scripture_source": "Ethics", "citation_ref": "Surah 4:34 vs Eph 5:25"}
        },

        # --- 1. MATHEMATICAL ERRORS (Inheritance Logic) ---
        {
            "id": "math_01",
            "text": "The Quranic inheritance laws in Surah 4:11-12 and 4:176 often result in a total sum that exceeds 100% (1.0). For example, if a man dies leaving three daughters, both parents, and a wife, the math is 2/3 + 1/3 + 1/8 = 1.125 (9/8). This mathematical impossibility proves the author was not the Creator of mathematics.",
            "meta": {"scripture_source": "Quranic Math", "citation_ref": "Surah 4:11-12"}
        },

        # --- 2. BORROWED PAGAN & ZOROASTRIAN MYTHS ---
        {
            "id": "pagan_01",
            "text": "The Quran's description of the 'Sirat Bridge' (a hair-thin bridge over hell) and the 'Hooris' (virgins in paradise) are almost identical to myths in the Zoroastrian 'Arda Viraf.' Additionally, the 5 daily prayers were a Zoroastrian practice long before Islam. This suggests Muhammad adapted local Persian religions rather than receiving new revelation.",
            "meta": {"scripture_source": "Comparative Religion", "citation_ref": "Zoroastrian Parallels"}
        },

        # --- 3. MORAL & SOCIAL CONTROVERSIES ---
        {
            "id": "moral_02",
            "text": "Sahih Muslim 1452 records the 'Adult Breastfeeding' (Rada' al-Kabir) incident where Muhammad told a woman to breastfeed a grown man so that he would become her 'relative' and be allowed to enter her house. This is widely considered a bizarre and morally problematic teaching compared to the modest, clear ethics of Jesus.",
            "meta": {"scripture_source": "Hadith", "citation_ref": "Sahih Muslim 1452"}
        },
        {
            "id": "moral_03",
            "text": "In Surah 66:1-5, Allah reveals verses specifically to settle a domestic dispute between Muhammad and his wives regarding his relation with his Coptic slave girl, Maria. Using 'Divine Revelation' to solve the Prophet's private marital problems suggests the Quran was self-serving rather than a universal message for mankind.",
            "meta": {"scripture_source": "Quran", "citation_ref": "Surah 66:1-5"}
        },

        # --- 4. THE CROSS VS THE SUBSTITUTION MYTH ---
        {
            "id": "cross_01",
            "text": "The Quran claims Jesus was not crucified but someone else was 'made to look like him' (Surah 4:157). This makes Allah the greatest deceiver, as he would have deceived all of Jesus' followers and the world for 600 years until Muhammad. The Bible says God cannot lie (Titus 1:2). Historical records from Thallus (AD 52) and Phlegon confirm the darkness during the crucifixion.",
            "meta": {"scripture_source": "Theology/History", "citation_ref": "Surah 4:157 vs Titus 1:2"}
        },

        # --- 5. LINGUISTIC ERRORS (Non-Arabic words) ---
        {
            "id": "lang_01",
            "text": "The Quran claims to be in 'pure Arabic' (Surah 16:103), yet it contains many foreign loanwords like 'Injil' (Greek), 'Al-Qistas' (Greek), 'Sijjil' (Persian), and 'Firdaws' (Persian/Greek). If Allah wrote it in pure Arabic, he would not need to borrow words from the 'People of the Book.'",
            "meta": {"scripture_source": "Linguistics", "citation_ref": "Surah 16:103 Analysis"}
        },

        # --- 6. PROPHECIES OF DESTRUCTION ---
        {
            "id": "prop_02",
            "text": "Muhammad predicted that the 'Last Hour' would come before a young boy of his time reached old age (Sahih Muslim 2953). That boy died over 1,300 years ago, and the Hour has not come. Deuteronomy 18:22 says that if a prophet speaks in the name of the Lord and the thing does not happen, he is a false prophet.",
            "meta": {"scripture_source": "Prophecy", "citation_ref": "Sahih Muslim 2953 vs Deut 18:22"}
        },

        # --- 1. HISTORICAL IMPOSSIBILITIES (The Al-Aqsa Error) ---
        {
            "id": "hist_04",
            "text": "Surah 17:1 claims Muhammad was taken to 'the farthest mosque' (Al-Aqsa) in Jerusalem. However, in AD 621, there was no mosque in Jerusalem; it was a Christian city (Aelia Capitolina), and the Temple Mount was a wasteland. The Al-Aqsa mosque was not built until roughly 70-80 years after Muhammad died. This is a massive historical anachronism.",
            "meta": {"scripture_source": "History", "citation_ref": "Surah 17:1"}
        },

        # --- 2. MUHAMMAD'S VULNERABILITY (The Magic Incident) ---
        {
            "id": "magic_01",
            "text": "Sahih al-Bukhari 5765 records that Muhammad was affected by black magic (sihr) cast by a Jew named Lubaid. It caused him to have hallucinations and think he had performed actions (like sexual relations with his wives) when he had not. A true prophet of the living God would not be under the power of a sorcerer. Jesus, however, had total authority over all demons and magic (Mark 1:27).",
            "meta": {"scripture_source": "Hadith", "citation_ref": "Sahih al-Bukhari 5765"}
        },

        # --- 3. BORROWING FROM JEWISH FABLES (The Mishnah) ---
        {
            "id": "myth_02",
            "text": "Surah 5:32 (the famous verse about saving one life) is not a divine revelation to Muhammad; it is a direct quote from the Jewish Mishnah Sanhedrin 4:5, written by Rabbis hundreds of years earlier. The Quran mistakenly presents a Rabbinic commentary as the Word of Allah.",
            "meta": {"scripture_source": "Comparative Religion", "citation_ref": "Surah 5:32 vs Mishnah Sanhedrin 4:5"}
        },

        # --- 4. TEXTUAL LOSS (The Stoning Verse) ---
        {
            "id": "text_03",
            "text": "Sahih al-Bukhari 6830 records Umar (the second Caliph) admitting that the Quran used to contain a verse about 'Stoning' (Rajam) for adultery, which is now missing. He stated that he would have added it back if he weren't afraid people would accuse him of changing the Quran. This proves the current Quran is incomplete.",
            "meta": {"scripture_source": "Hadith", "citation_ref": "Sahih al-Bukhari 6830"}
        },

        # --- 5. THE 'AL-LAT' CONTRADICTION ---
        {
            "id": "pagan_02",
            "text": "The name 'Allah' was used by pagan Arabs for a high god long before Muhammad, and his own father was named 'Abd-Allah' (Slave of Allah) while still a pagan. This shows Muhammad did not introduce a new God, but adapted a local Meccan deity. Christianity introduces the Father, Son, and Holy Spirit—a unique revelation unknown to pagans.",
            "meta": {"scripture_source": "History/Etymology", "citation_ref": "Pre-Islamic Archaeology"}
        },

        # --- 6. MORAL CONTRAST (Forgiveness vs. Vengeance) ---
        {
            "id": "ethic_02",
            "text": "In Sahih al-Bukhari 3018, Muhammad said: 'I have been ordered to fight the people till they say La ilaha illallah.' In contrast, Jesus said in Matthew 26:52, 'Put your sword back into its place. For all who take the sword will perish by the sword.' The spread of Islam was by compulsion; the spread of Christianity was by the conviction of the Holy Spirit.",
            "meta": {"scripture_source": "Comparative Ethics", "citation_ref": "Bukhari 3018 vs Matt 26:52"}
        },

        # --- 1. DEBUNKING PARTIALISM (The "One-Third" Error) ---
        {
            "id": "trinity_01",
            "text": "The Trinity is not a division of God into parts (Partialism). In mathematics, it is not 1+1+1=3, but 1x1x1=1. God is 'Simple' (Divine Simplicity), meaning He has no parts. The Father, Son, and Spirit each possess the 100% fullness of the Divine Essence (Ousia) simultaneously. Jesus is not 33% of God; He is 'the fullness of the Deity in bodily form' (Colossians 2:9).",
            "meta": {"scripture_source": "Systematic Theology", "citation_ref": "Colossians 2:9 / Athanasian Creed"}
        },

        # --- 2. THE ETERNAL SONSHIP (Countering the 'Physical Son' Myth) ---
        {
            "id": "trinity_02",
            "text": "The Quran in Surah 6:101 claims God cannot have a son because He has no consort (wife). This is a 'Category Error.' Christians do not believe in physical procreation (begetting). The 'Sonship' of Jesus refers to His eternal relationship as the 'Logos' (Word/Reason) of the Father. Just as a mind generates a thought without a 'wife,' the Father eternally generates the Son (John 1:1).",
            "meta": {"scripture_source": "Theology", "citation_ref": "John 1:1 / Surah 6:101 Error"}
        },

        # --- 3. THE DEITY OF THE HOLY SPIRIT ---
        {
            "id": "trinity_03",
            "text": "Islam often claims the Holy Spirit is just the Angel Gabriel. However, the Bible identifies the Holy Spirit as God. In Acts 5:3-4, Peter tells Ananias he lied to the Holy Spirit and then says, 'You have not lied to men but to God.' The Spirit possesses divine attributes: Omniscience (1 Cor 2:10-11) and Omnipresence (Psalm 139:7).",
            "meta": {"scripture_source": "Bible", "citation_ref": "Acts 5:3-4, 1 Cor 2:10"}
        },

        # --- 4. THE BAPTISM OF CHRIST (The Manifestation) ---
        {
            "id": "trinity_04",
            "text": "At the Baptism of Jesus (Matthew 3:16-17), all three Persons are present simultaneously: The Son is in the water, the Holy Spirit descends like a dove, and the Father speaks from heaven. This proves they are distinct Persons (Hypostasis) but one in action and purpose, refuting the Islamic claim that Christians 'invented' the Trinity later.",
            "meta": {"scripture_source": "Gospel", "citation_ref": "Matthew 3:16-17"}
        },

        # --- 5. LOGICAL ANALOGIES (The Nature of Fire) ---
        {
            "id": "trinity_05",
            "text": "To explain the Trinity to skeptics: Consider fire. A flame has light, heat, and the flame itself. They are distinct (you can feel heat without seeing light), yet they are one fire. If you remove one, you no longer have a flame. Similarly, the Father, Son, and Spirit are distinct in Relation but inseparable in Essence.",
            "meta": {"scripture_source": "Apologetics", "citation_ref": "Early Church Fathers"}
        },

        # --- 6. THE 'I AM' REVELATION ---
        {
            "id": "trinity_06",
            "text": "In John 8:58, Jesus says, 'Before Abraham was, I AM.' He uses the divine name of God revealed to Moses in Exodus 3:14 (Ehyeh). The Jews understood He was claiming to be the self-existent God and tried to stone Him. This proves Jesus did not just claim to be a prophet, but the eternal God of Israel.",
            "meta": {"scripture_source": "Bible", "citation_ref": "John 8:58 / Exodus 3:14"}
        },

        # --- 1. THE DEAD SEA SCROLLS (Old Testament Proof) ---
        {
            "id": "mss_01",
            "text": "The Dead Sea Scrolls (discovered 1947) contain the 'Great Isaiah Scroll' dating to c. 125 BC. This is 1,000 years older than the previous oldest manuscript (Masoretic Text). The text is 95% identical, with the 5% difference being minor spelling variations. This proves the Old Testament was perfectly preserved long before Islam existed.",
            "meta": {"scripture_source": "Archaeology", "citation_ref": "Great Isaiah Scroll (1QIs-a)"}
        },

        # --- 2. NEW TESTAMENT VOLUME (The 'Numbers' Argument) ---
        {
            "id": "mss_02",
            "text": "There are over 5,800 Greek New Testament manuscripts, plus 10,000 Latin and 9,000 other versions. In contrast, Homer’s Iliad has only 643. If we doubt the Bible's preservation, we must doubt all of ancient history. We have manuscripts like P52 (John's Gospel) dating to within 30-50 years of the original writing.",
            "meta": {"scripture_source": "Textual Criticism", "citation_ref": "P52 Papyrus / Bodmer Papyri"}
        },

        # --- 3. THE EARLY CHURCH FATHERS (The 'Recovery' Proof) ---
        {
            "id": "mss_03",
            "text": "Even if every single Bible manuscript were destroyed, we could reconstruct 99% of the New Testament just from the quotes of the Early Church Fathers (like Ignatius, Polycarp, and Clement) who wrote in the 1st and 2nd centuries. This proves the message was fixed and widespread from the very beginning.",
            "meta": {"scripture_source": "Patristics", "citation_ref": "Clement of Rome (AD 95)"}
        },

        # --- 4. THE QURANIC MANUSCRIPT PROBLEM (Sana'a Manuscripts) ---
        {
            "id": "mss_04",
            "text": "The Sana'a manuscripts found in Yemen show 'palimpsests' (erased and rewritten text) where the lower layer of the Quran has a different version of the verses than the upper layer. This proves the Quran underwent a process of editing and change, contradicting the claim that it was never altered.",
            "meta": {"scripture_source": "Islamic Archaeology", "citation_ref": "Sana'a Palimpsest (DAM 01-27.1)"}
        },

        # --- 5. THE 'TAHRIF' CONTRADICTION ---
        {
            "id": "mss_05",
            "text": "The Quran never actually says the physical text of the Bible was corrupted; it says some people 'twisted it with their tongues' (Surah 3:71). If the Bible was physically corrupted, then Allah failed to protect His Word, which contradicts Surah 6:115: 'None can change His words.'",
            "meta": {"scripture_source": "Internal Logic", "citation_ref": "Surah 3:71 vs 6:115"}
        }
    ]

    # Prepare for upload
    ids = [item["id"] for item in data]
    texts = [item["text"] for item in data]
    metadatas = [item["meta"] for item in data]
    
    # Generate embeddings
    print("Generating embeddings for ammunition...")
    embeddings_list = embeddings.embed_documents(texts)

    # Upsert into Chroma
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=texts
    )
    print(f"Successfully added {len(ids)} apologetic records to {CHROMA_COLLECTION_NAME}.")

if __name__ == "__main__":
    inject_apologetics()